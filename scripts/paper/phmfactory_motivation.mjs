/** Regenerate the PHMFactory motivation schematic. No data or fitted result is drawn.
 * Usage: node scripts/paper/phmfactory_motivation.mjs /path/to/motivation.svg
 * Each concept is a named SVG group; labels remain editable SVG text.
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
const output = process.argv[2];
if (!output || process.argv.length !== 3) {
  console.error('Usage: node phmfactory_motivation.mjs <output.svg>');
  process.exit(2);
}
const s = [];
const esc = x => String(x).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
function text(x, y, lines, cls = 'body', anchor = 'start') {
  const a = Array.isArray(lines) ? lines : [lines];
  s.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${a.map((v,i) => `<tspan x="${x}" dy="${i ? 31 : 0}">${esc(v)}</tspan>`).join('')}</text>`);
}
function box(id, x, y, w, h, lines, cls = 'box') {
  s.push(`<g id="${id}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" class="${cls}"/>`);
  const n = Array.isArray(lines) ? lines.length : 1;
  text(x + w/2, y + h/2 - (n-1)*15.5 + 7, lines, 'body', 'middle');
  s.push('</g>');
}
function arrow(id, x1, y1, x2, y2, dashed = false) {
  s.push(`<path id="${id}" d="M ${x1} ${y1} L ${x2} ${y2}" class="arrow${dashed ? ' dashed' : ''}" marker-end="url(#arrowhead)"/>`);
}
function panel(id, x, y, label, title) {
  s.push(`<g id="${id}"><rect x="${x}" y="${y}" width="850" height="350" rx="12" class="panel"/>`);
  text(x+25, y+39, label, 'label'); text(x+65, y+39, title, 'heading');
}
s.push(`<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="850" viewBox="0 0 1800 850" role="img" aria-labelledby="title desc">
<title id="title">PHMFactory: from reusable components to compositional task execution</title>
<desc id="desc">Existing reusable data, model and task interfaces do not by themselves quantify extension locality. The proposed paired comparison changes assembly while holding numerical components, protocol and budget fixed. Outcomes are change propagation, valid unattended completion, prediction agreement and cost. This is a study design, not empirical results.</desc>
<defs><marker id="arrowhead" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L10,4 L0,8 Z" fill="#526170"/></marker></defs>
<style>
text { font-family: 'DejaVu Sans', sans-serif; fill: #202b36; }
.title { font-size: 34px; font-weight: 600; }
.heading { font-size: 28px; font-weight: 600; }
.label { font-size: 28px; font-weight: 700; }
.body { font-size: 26px; }
.small { font-size: 24px; fill: #526170; }
.panel { fill: #ffffff; stroke: #bac4ce; stroke-width: 1.5; }
.box { fill: #f4f6f8; stroke: #98a6b4; stroke-width: 1.5; }
.accent { fill: #eaf2f8; stroke: #4b789b; stroke-width: 2; }
.arrow { stroke: #526170; fill: none; stroke-width: 2; }
.dashed { stroke-dasharray: 7 5; }
</style><rect width="1800" height="850" fill="white"/>`);
text(40, 46, 'From reusable components to compositional PHM task execution', 'title');
panel('a-established-capability', 40, 80, 'a', 'Existing modular components');
box('data-choices', 70, 154, 235, 110, ['Data D', 'units / domains']);
box('model-choices', 346, 154, 235, 110, ['Model M', 'type / weights']);
box('task-choices', 622, 154, 235, 110, ['Task T', 'labels / loss']);
arrow('data-to-request',187,267,187,300); arrow('model-to-request',463,267,463,300); arrow('task-to-request',739,267,739,300);
box('request', 70, 307, 787, 63, 'Declared components, protocol and budget');
text(70, 407, 'Task changes may require different targets and prediction heads.', 'small');
s.push('</g>');
panel('b-extension-question', 910, 80, 'b', 'Extension: local or coupled?');
box('extension', 940, 153, 240, 85, ['Add / replace', 'component'], 'accent');
box('other-owners', 1235, 153, 490, 85, ['Other components', 'and orchestration']);
arrow('possible-propagation',1186,196,1227,196,true);
box('admissible-space', 940, 279, 785, 77, ['Admissible combinations Ω_adm', 'not every data–model–task combination']);
text(940, 398, 'Required changes S(e) vs. observed changes A(e).', 'small');
s.push('</g>');
panel('c-controlled-intervention', 40, 460, 'c', 'Control: change assembly only');
box('fixed-numerical-experiment',70,533,787,64,'Same numerical experiment and budget');
arrow('fixed-to-control',248,600,248,637); arrow('fixed-to-treatment',664,600,664,637);
box('control-direct',70,644,355,76,['Direct assembly', 'reference execution']);
box('treatment-factory',502,644,355,76,['Factory assembly', 'shared execution'],'accent');
text(70, 774, 'Paired extensions; same model computation and protocol.', 'small');
s.push('</g>');
panel('d-observable-outcomes',910,460,'d','Outcomes: test the automation claim');
box('outcome-locality',940,539,375,76,['Change propagation', 'off-target changes I(e)']);
box('outcome-completion',1350,539,375,76,['Valid completion', 'unattended / requested']);
box('outcome-agreement',940,641,375,76,['Prediction agreement', 'declared tolerance ε']);
box('outcome-cost',1350,641,375,76,['Time and memory', 'human intervention']);
text(940,774,'Compatible and incompatible requests; no empirical values shown.','small');
s.push('</g></svg>');
mkdirSync(dirname(resolve(output)), { recursive: true });
writeFileSync(output, s.join('\n') + '\n', 'utf8');
console.log(resolve(output));
