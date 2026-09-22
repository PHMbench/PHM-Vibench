"""Fit95 tests on explicit generated observations, not industrial results.

Component tests use a small test network. NativeFit95Integration additionally
runs the real Data/Model/Task/Trainer lifecycle on temporary CPU sine fixtures.
Neither validates natural-acquisition eligibility or an industrial CUDA run.
"""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from scripts.tii_one_model import constraints, fit95_constraints, acceptance_constraints, main
from src.task_factory.Components.tii_evaluation import evaluate_source_predictions, source_fit95_report
from src.task_factory.Components.tii_fit95 import SourceFit95, FIT95_STEPS, retain_predictions
from src.task_factory.task.DG.tii_joint import task as JointTask


def config():
    return dict(pipeline='Pipeline_01_Fault_Diagnosis',
        environment=dict(iterations=1,seed=0,wandb=False,swanlab=False),
        data=dict(evidence_kind='natural_acquisition',use_cache=True,normalization='source_rms',
                  source_batch_size=32,seed=0,rounds=10000),
        model=dict(type='ISFM',name='M_01_ISFM',embedding='SupportConditionedTokenizer',
                   token_organization='support',backbone='B_04_Dlinear',task_head='H_01_Linear_cla'),
        task=dict(type='DG',name='tii_joint',sampling_seed=0,lambda_common=0,lambda_private=0,
                  source_system_ids=[1,2],target_system_id=[1,2]),
        trainer=dict(num_epochs=1,devices=1,device='cuda',test_after_fit=False,
                     early_stopping=False,val_check_interval=100,num_sanity_val_steps=0))


def observations(root, errors=None, *, imbalance=False):
    """Independent expected population and explicit deterministic test predictions."""
    classes={'1':[0,1], '2':[0,1]}
    expected=[]
    for source in classes:
        for role in ('source_train','source_val'):
            for i in range(20):
                expected.append(dict(dataset=source,recording_id=f'{role}_{i}',
                    channel='0',window_start=0,window_end=16,group=f'{role}_{i//10}',
                    role=role,file_id=f'{source}_{role}_{i}',true_label=0 if imbalance and i<19 else i%2))
    expected=pd.DataFrame(expected)
    expected.to_csv(root/'expected_source_windows.csv',index=False)
    (root/'local_class_map.json').write_text(json.dumps(classes))

    def predictions(role, checkpoint, errors_by_source=None):
        rows=expected[expected.role==role].copy()
        logits=[]; probs=[]
        for source, part in rows.groupby('dataset',sort=False):
            count=(errors_by_source or {}).get(source,0)
            for i, row in enumerate(part.itertuples()):
                pred=1-row.true_label if i<count else row.true_label
                x=np.array([3.,-3.]) if pred==0 else np.array([-3.,3.])
                p=np.exp(x-x.max());p/=p.sum()
                logits.append(json.dumps(x.tolist()));probs.append(json.dumps(p.tolist()))
        rows['logits']=logits;rows['probabilities']=probs
        rows['checkpoint']=str(checkpoint);rows['method']='support';rows['seed']=0;rows['intervention']='full'
        return rows
    predictions('source_train','selected.ckpt',errors).to_csv(root/'source_training_predictions.csv',index=False)
    return expected,classes,predictions


def complete_run(root, *, selected_errors=None, terminal_errors=None):
    _,classes,predict=observations(root,selected_errors)
    (root/'run.json').write_text(json.dumps(dict(mode='one_model_fit95',model_fits_completed=1,
        source_system_ids=[1,2],best_checkpoint='selected.ckpt')))
    (root/'fit95').mkdir()
    curve=[]
    for step in FIT95_STEPS:
        stem=root/'fit95'/f'step_{step:05d}'
        for role in ('source_train','source_val'):
            file=Path(str(stem)+f'_{role}.csv')
            errors=terminal_errors if step==10000 and role=='source_train' else None
            predict(role,str(stem)+'.ckpt',errors).to_csv(file,index=False)
            _,_,metrics=evaluate_source_predictions(file,expected_windows_file=root/'expected_source_windows.csv',
                local_class_map=classes,checkpoint=str(stem)+'.ckpt',intervention='full',role=role)
            metrics['global_step']=step;metrics['role']=role;curve.append(metrics)
    pd.concat(curve,ignore_index=True).to_csv(root/'source_fit_curve.csv',index=False)


class Fit95Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def evaluate(self,role='source_train'):
        return evaluate_source_predictions(self.root/'source_training_predictions.csv',
            expected_windows_file=self.root/'expected_source_windows.csv',
            local_class_map={'1':[0,1],'2':[0,1]},checkpoint='selected.ckpt',intervention='full',role=role)
    def test_fixed_fit95_budget(self):fit95_constraints(config())
    def test_legacy_twenty_round_mode(self):
        c=config();c['data']['rounds']=20;c['trainer']['val_check_interval']=10;acceptance_constraints(c)
    def test_short_budget_rejected(self):
        c=config();c['data']['rounds']=20
        with self.assertRaises(ValueError):fit95_constraints(c)
    def test_wrong_backbone_rejected_not_silently_changed(self):
        c=config();c['model']['backbone']='Transformer'
        with self.assertRaises(ValueError):fit95_constraints(c)
    def test_no_extra_model_seed_or_dataset(self):
        for section,key,value in [('environment','iterations',2),('environment','seed',1),
                                  ('task','source_system_ids',[1,2,3,4,5,6])]:
            with self.subTest(key=key):
                c=config();c[section][key]=value
                with self.assertRaises(ValueError):fit95_constraints(c)
    def test_train_role_explicit_and_correct(self):
        observations(self.root,{'1':1,'2':0})
        _,_,m=self.evaluate();self.assertAlmostEqual(m.iloc[0].accuracy,.95)
        self.assertTrue(m.scope.eq('training_resubstitution').all())
    def test_train_not_accepted_as_validation(self):
        observations(self.root)
        with self.assertRaises(ValueError):self.evaluate('source_val')
    def test_omitted_training_window_rejected(self):
        observations(self.root)
        p=self.root/'source_training_predictions.csv';t=pd.read_csv(p);t.iloc[1:].to_csv(p,index=False)
        with self.assertRaises(ValueError):self.evaluate()
    def test_duplicate_training_window_rejected(self):
        observations(self.root)
        p=self.root/'source_training_predictions.csv';t=pd.read_csv(p);pd.concat([t,t.iloc[:1]]).to_csv(p,index=False)
        with self.assertRaises(ValueError):self.evaluate()
    def test_cross_role_group_leakage_rejected(self):
        observations(self.root)
        p=self.root/'expected_source_windows.csv';t=pd.read_csv(p);t.loc[t.role=='source_val','group']='source_train_0';t.to_csv(p,index=False)
        with self.assertRaises(ValueError):self.evaluate()
    def test_missing_group_rejected(self):
        observations(self.root)
        p=self.root/'source_training_predictions.csv';t=pd.read_csv(p);t.loc[0,'group']=np.nan;t.to_csv(p,index=False)
        with self.assertRaises(ValueError):self.evaluate()
    def test_checkpoint_mixing_rejected(self):
        observations(self.root)
        p=self.root/'source_training_predictions.csv';t=pd.read_csv(p);t.loc[0,'checkpoint']='per_source_best.ckpt';t.to_csv(p,index=False)
        with self.assertRaises(ValueError):self.evaluate()
    def test_every_source_not_average(self):
        complete_run(self.root,selected_errors={'1':0,'2':2}) # Mean=.95, worst=.90.
        result=source_fit95_report(self.root)
        self.assertFalse(result['source_fit_target_met'])
        self.assertAlmostEqual(result['min_source_train_group_accuracy'],.9)
    def test_exact_threshold_passes(self):
        complete_run(self.root,selected_errors={'1':1,'2':1})
        self.assertTrue(source_fit95_report(self.root)['source_fit_target_met'])
    def test_terminal_not_selected(self):
        complete_run(self.root,selected_errors={'1':2},terminal_errors=None)
        r=source_fit95_report(self.root)
        self.assertFalse(r['source_fit_target_met']);self.assertTrue(r['terminal_source_fit_target_met'])
    def test_missing_curve_point_rejected(self):
        complete_run(self.root)
        p=self.root/'source_fit_curve.csv';pd.read_csv(p).iloc[1:].to_csv(p,index=False)
        with self.assertRaises(ValueError):source_fit95_report(self.root)
    def test_curve_tampering_rejected(self):
        complete_run(self.root)
        p=self.root/'source_fit_curve.csv';t=pd.read_csv(p);t.loc[0,'accuracy']=.99;t.to_csv(p,index=False)
        with self.assertRaises(ValueError):source_fit95_report(self.root)
    def test_curve_raw_missing_rejected(self):
        complete_run(self.root);(self.root/'fit95/step_00020_source_train.csv').unlink()
        with self.assertRaises(ValueError):source_fit95_report(self.root)
    def test_nonfinite_prediction_rejected(self):
        observations(self.root)
        p=self.root/'source_training_predictions.csv';t=pd.read_csv(p);t.loc[0,'logits']='[NaN, 0]';t.to_csv(p,index=False)
        with self.assertRaises(ValueError):self.evaluate()
    def test_prediction_recovery_does_not_overwrite(self):
        _,_,predict=observations(self.root)
        path=self.root/'retained.csv';first=predict('source_train','x.ckpt');retain_predictions(path,first)
        before=path.read_bytes();retain_predictions(path,first)
        with self.assertRaises(ValueError):retain_predictions(path,predict('source_train','x.ckpt',{'1':1}))
        self.assertEqual(before,path.read_bytes())
    def test_imbalance_not_hidden_by_accuracy(self):
        _,_,predict=observations(self.root,imbalance=True)
        t=predict('source_train','selected.ckpt')
        x=np.array([3.,-3.]);p=np.exp(x-x.max());p/=p.sum()
        t['logits']=json.dumps(x.tolist());t['probabilities']=json.dumps(p.tolist());t.to_csv(self.root/'source_training_predictions.csv',index=False)
        _,_,m=self.evaluate()
        self.assertTrue(np.allclose(m.window_accuracy,.95));self.assertTrue(np.allclose(m.balanced_accuracy,.5))
        self.assertTrue(np.allclose(m.majority_accuracy,.95))
    def test_no_h5_or_model_in_offline_evaluator(self):
        import inspect
        from src.task_factory.Components import tii_evaluation
        text=inspect.getsource(tii_evaluation)
        self.assertNotIn('import torch',text);self.assertNotIn('build_model',text);self.assertNotIn('build_data',text)
    def test_analyze_returns_failure_if_target_not_met(self):
        with patch('src.task_factory.Components.tii_evaluation.analyze_source_acceptance',return_value={'source_fit_target_met':False}):
            self.assertEqual(main(['--analyze-only','--output',str(self.root)]),3)


class TinyNet(torch.nn.Module):
    """Explicit test network, not the industrial M_01_ISFM."""
    def __init__(self,metadata):
        super().__init__();self.metadata=metadata
        self.embedding=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.SiLU(),torch.nn.Dropout(.2))
        self.task_head=torch.nn.Module();self.task_head.mutiple_fc=torch.nn.ModuleDict({str(s):torch.nn.Linear(4,2) for s in [1,2]})
    def forward(self,x,file_id,task_id,incremental,availability):
        source=self.metadata[file_id[0]]['Dataset_id']
        return self.task_head.mutiple_fc[str(source)](self.embedding(x.mean(1)))


def task_fixture(root):
    metadata={};sources={};windows=[];val=[]
    for s in [1,2]:
        for role in ['source_train','source_val']:
            n=4;fid=[f'{s}_{role}_{i}' for i in range(n)];rec=list(fid)
            for i,f in enumerate(fid):metadata[f]={'Dataset_id':s,'Label':i%2}
            item=dict(x=torch.tensor([[[1.,0.]],[[0.,1.]],[[1.,0.]],[[0.,1.]]]),
                incremental=torch.zeros(n,1,2),availability=torch.zeros(n),y=torch.tensor([0,1,0,1]),
                file_id=fid,recording_id=rec,group=[f'{role}_{i//2}' for i in range(n)],role=[role]*n)
            if role=='source_train':sources[s]=item
            else:val.append(dict(item,source=s))
            for i in range(n):windows.append(dict(dataset=s,recording_id=rec[i],group=item['group'][i],role=role,
                file_id=fid[i],channel=0,window_start=0,window_end=2,true_label=i%2))
    factory=NS(window_inventory=[{k:v for k,v in w.items() if k!='true_label'} for w in windows],
               train_dataset=NS(sources=sources),val_dataset=val)
    pd.DataFrame(windows).to_csv(root/'expected_source_windows.csv',index=False)
    (root/'local_class_map.json').write_text(json.dumps({'1':[0,1],'2':[0,1]}))
    net=TinyNet(metadata)
    taskargs=NS(loss='CE',optimizer='adamw',lambda_common=0,lambda_private=0,scheduler=None,
                metrics=['acc'],source_system_ids=[1,2],lr=.001,weight_decay=.0001,acceptance_audit=False)
    task=JointTask(net,NS(source_batch_size=4),NS(),taskargs,NS(),NS(),metadata)
    return factory,task


class TaskObservationTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_full_training_export_not_sampler(self):
        factory,task=task_fixture(self.root)
        exported=task.source_predictions(factory,'unit.ckpt',role='source_train')
        self.assertEqual(len(exported),8);self.assertEqual(set(exported.role),{'source_train'})
    def test_legacy_validation_export_still_works(self):
        factory,task=task_fixture(self.root)
        self.assertEqual(len(task.source_validation_predictions(factory,'unit.ckpt')),8)
    def test_target_export_rejected(self):
        factory,task=task_fixture(self.root)
        with self.assertRaises(ValueError):task.source_predictions(factory,'unit.ckpt',role='query')
    def test_observation_restores_mode_and_rng(self):
        factory,task=task_fixture(self.root);task.train()
        callback=SourceFit95(factory,self.root)
        trainer=NS(global_step=0,save_checkpoint=lambda p:Path(p).write_text('test checkpoint marker'))
        state=torch.random.get_rng_state().clone()
        callback.on_train_start(trainer,task)
        self.assertTrue(task.training);self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        curve=pd.read_csv(self.root/'source_fit_curve.csv');self.assertEqual(len(curve),4)
        callback.on_train_start(trainer,task);self.assertEqual(len(pd.read_csv(self.root/'source_fit_curve.csv')),4)
    def test_real_lightning_optimizer_trajectory_unchanged(self):
        states=[]
        for observe in [False,True]:
            run=self.root/str(observe);run.mkdir();torch.manual_seed(42)
            factory,task=task_fixture(run)
            train=[factory.train_dataset.sources]*3
            callbacks=[SourceFit95(factory,run)] if observe else []
            trainer=pl.Trainer(max_epochs=1,accelerator='cpu',devices=1,logger=False,enable_checkpointing=False,
                enable_progress_bar=False,enable_model_summary=False,limit_val_batches=0,
                num_sanity_val_steps=0,callbacks=callbacks,default_root_dir=str(run))
            with patch('src.task_factory.Components.tii_fit95.FIT95_STEPS',(0,1,3)):
                trainer.fit(task,train_dataloaders=DataLoader(train,batch_size=None))
            self.assertEqual(trainer.global_step,3)
            states.append({k:v.clone() for k,v in task.state_dict().items()})
        self.assertEqual(set(states[0]),set(states[1]))
        for key in states[0]:self.assertTrue(torch.equal(states[0][key],states[1][key]),key)


class ShellTests(unittest.TestCase):
    """Command-recorder tests only: they do not train or invoke CUDA."""
    def call_script(self, arguments, failure=''):
        import os, subprocess
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);fake=root/'conda'
        fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n'
                        'if [[ -n "${FAIL_MATCH:-}" && "$*" == *"$FAIL_MATCH"* ]]; then exit 3; fi\n')
        fake.chmod(0o755)
        env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],CALL_LOG=str(root/'calls'),FAIL_MATCH=failure)
        script=Path(__file__).resolve().parents[1]/'scripts/run_tii_fit95.sh'
        result=subprocess.run(['bash',str(script),*arguments],env=env,text=True,capture_output=True)
        lines=(root/'calls').read_text().splitlines() if (root/'calls').exists() else []
        return result,lines
    def test_shell_one_fit_then_csv_plot_and_analysis(self):
        r,lines=self.call_script(['config.yaml','output'])
        self.assertEqual(r.returncode,0);self.assertEqual(len(lines),3)
        self.assertEqual(sum('--fit95' in s for s in lines),1)
        self.assertIn('scripts.tii_plot_fit95',lines[1]);self.assertIn('--analyze-only',lines[2])
    def test_shell_check_only_never_fits(self):
        r,lines=self.call_script(['--check-only','config.yaml'])
        self.assertEqual(r.returncode,0);self.assertEqual(len(lines),1);self.assertIn('--check-only',lines[0])
    def test_shell_reuse_never_fits(self):
        r,lines=self.call_script(['--reuse','output'])
        self.assertEqual(r.returncode,0);self.assertEqual(len(lines),3)
        self.assertIn('--export-only',lines[0]);self.assertFalse(any('--fit95' in s for s in lines))
    def test_shell_training_failure_stops_no_retry(self):
        r,lines=self.call_script(['config.yaml','output'],failure='--fit95')
        self.assertEqual(r.returncode,3);self.assertEqual(len(lines),1)
    def test_shell_target_failure_after_plots_preserved(self):
        r,lines=self.call_script(['config.yaml','output'],failure='--analyze-only')
        self.assertEqual(r.returncode,3);self.assertEqual(len(lines),3)


class NativeFit95Integration(unittest.TestCase):
    def test_native_fixed_budget_fit_export_and_offline_reuse(self):
        """Real Factory lifecycle on generated sine data, CPU; not industrial evidence."""
        import subprocess
        import sys
        import yaml
        from scripts import tii_one_model as entry
        from scripts.tii_plot_fit95 import main as plot
        from phmfactory.config import analyze_config
        from src.runtime.classification import ClassificationHooks, run_classification_pipeline

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'explicit_tensor_fixture'
            subprocess.run([sys.executable, '-m', 'scripts.tii_make_fixture', '--output', str(inputs)],
                cwd=entry.ROOT, check=True, capture_output=True, text=True)
            c = yaml.safe_load((inputs/'native.yaml').read_text())
            c['model']['token_organization'] = 'support'
            c['data']['rounds'] = 10000
            c['trainer']['val_check_interval'] = 100
            output = root/'fit95'; output.mkdir()
            c['environment']['output_dir'] = str(root/'native_results')
            c['task'].update(acceptance_audit=True, acceptance_audit_rounds=20,
                             acceptance_output=str(output/'audit'))
            runtime = output/'runtime.yaml'; runtime.write_text(yaml.safe_dump(c, sort_keys=False))
            # Tests use explicit CPU fixtures via the native lifecycle; the
            # production command still rejects this evidence/device regime.
            with self.assertRaisesRegex(ValueError, 'tensor_fixture'):
                entry.fit95_constraints(c)
            compiled = analyze_config(runtime)

            class Capture(ClassificationHooks):
                context = None
                def after_stack_built(self, context):
                    if self.context is not None:
                        raise AssertionError('more than one native model')
                    self.context = context
                    entry._expected_source_windows(context.data_factory).to_csv(
                        output/'expected_source_windows.csv', index=False)
                    classes = {str(k): list(range(context.model.task_head.mutiple_fc[str(k)].out_features))
                               for k in context.task.sources}
                    (output/'local_class_map.json').write_text(json.dumps(classes))
                    context.trainer.callbacks.append(SourceFit95(context.data_factory, output))

            capture = Capture()
            result = run_classification_pipeline(NS(config_path=str(runtime),
                compiled_run_spec=compiled, resolved_pipeline=compiled.pipeline), hooks=capture)
            self.assertEqual(result['status'], 'succeeded')
            self.assertEqual(capture.context.trainer.global_step, 10000)
            self.assertEqual(len(result['best_checkpoints']), 1)
            report = dict(mode='one_model_fit95', evidence_kind='tensor_fixture_not_industrial',
                model_fits_completed=1, rounds=10000, source_system_ids=list(capture.context.task.sources),
                best_checkpoint=result['best_checkpoints'][0], status='training_completed_export_pending')
            entry._save_report(output, report)
            entry._export_selected(capture.context, output, report)
            self.assertEqual(report['status'], 'completed_source_fit95_diagnostics_not_transfer')
            self.assertEqual(report['analysis']['actual_joint_updates'], 10000)
            self.assertEqual(report['analysis']['gradient_audited_updates'], 20)
            self.assertIsNone(report['analysis']['transfer_delta'])
            expected = pd.read_csv(output/'expected_source_windows.csv')
            exported = pd.read_csv(output/'source_training_predictions.csv')
            self.assertEqual(len(exported), int((expected.role == 'source_train').sum()))
            curve = pd.read_csv(output/'source_fit_curve.csv')
            self.assertEqual(set(curve.global_step), set(FIT95_STEPS))
            with patch.object(entry, '_gpu0', side_effect=AssertionError('offline reuse accessed GPU')):
                self.assertEqual(entry.export_completed_acceptance(output), 0)
            plot(['--output', str(output)])
            for name in (f'source_{source}_{metric}' for source in capture.context.task.sources
                         for metric in ('accuracy', 'nll')):
                for suffix in ('svg','pdf','png'):
                    self.assertGreater((output/'fit_figures'/f'{name}.{suffix}').stat().st_size, 0)
                svg = (output/'fit_figures'/f'{name}.svg').read_text()
                self.assertIn('<text', svg); self.assertNotIn('<image', svg)


if __name__=='__main__':
    unittest.main()
