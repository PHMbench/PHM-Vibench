# UXFD Low-Tier Source Audit

Status: source-hygiene triage only. This report is not citation replacement evidence.

- Ready: `True`
- Root: `paper/UXFD_paper`
- Findings: `263`
- Blockers: `0`
- Triage-only findings: `263`

| Severity | Paper | Marker | Location | Text |
|---|---|---|---|---|
| `triage` | `MOE_explainable` | `MDPI` | `paper/UXFD_paper/MOE_explainable/doc/11-19/brainstorm.md:60` | * 用 SHAP 分析振动特征对故障分类贡献度，找出如偏度、波形因子等关键统计量。([MDPI][9]) |
| `triage` | `MOE_explainable` | `MDPI` | `paper/UXFD_paper/MOE_explainable/doc/11-19/brainstorm.md:145` | * **Neural Additive Model (NAM)**：每个特征一个子网络 (f_j(z_j))，总路由 logit 是 (\sum_j f_j(z_j))，可以画曲线看「某个物理特征改变对路由概率的影响」。([MDPI][9]) |
| `triage` | `MOE_explainable` | `MDPI` | `paper/UXFD_paper/MOE_explainable/doc/11-19/brainstorm.md:224` | 这和 MoE-KAN/IME 里分析 expert 对不同模式分工的做法类似。([MDPI][13]) |
| `triage` | `MOE_explainable` | `MDPI` | `paper/UXFD_paper/MOE_explainable/doc/11-19/brainstorm.md:259` | 借鉴现有故障诊断里用 SHAP 分析特征贡献的做法：([MDPI][9]) |
| `triage` | `MOE_explainable` | `MDPI` | `paper/UXFD_paper/MOE_explainable/doc/11-19/brainstorm.md:277` | 结合 KAN & 可解释 MoE 的思路：([MDPI][14]) |
| `triage` | `MOE_explainable` | `MDPI` | `paper/UXFD_paper/MOE_explainable/doc/11-19/brainstorm.md:392` | [9]: https://www.mdpi.com/2076-3417/13/4/2038?utm_source=chatgpt.com "Explainable AI for Machine Fault Diagnosis: Understanding Features’ Contribution in Machine Learning Models for Industrial Condition Monitoring" |
| `triage` | `MOE_explainable` | `MDPI` | `paper/UXFD_paper/MOE_explainable/doc/11-19/brainstorm.md:396` | [13]: https://www.mdpi.com/2079-9292/13/20/4116?utm_source=chatgpt.com "Interpretable Mixture of Experts for Decomposition Network on Server Performance Metrics Forecasting" |
| `triage` | `MOE_explainable` | `MDPI` | `paper/UXFD_paper/MOE_explainable/doc/11-19/brainstorm.md:397` | [14]: https://www.mdpi.com/1099-4300/27/4/403?utm_source=chatgpt.com "Explainable Fault Classification and Severity Diagnosis in Rotating Machinery Using Kolmogorov–Arnold Networks" |
| `triage` | `Paper_fuzzy_XFD` | `IEEE TIM` | `paper/UXFD_paper/Paper_fuzzy_XFD/plan/12_14/codex/plan_fuzzy_xfd_12_14.md:410` | - [ ] 论文初稿完成（目标期刊：IEEE TII/IEEE TIM） |
| `triage` | `TII_operator_attention` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/TII_operator_attention/doc/修改意见与计划_无新实验版.md:128` | 2. **IEEE Transactions on Instrumentation and Measurement** |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/prompt/文献综述prompt.md:25` | 1. **检索数据库/来源优先级（国际）**：IEEE Xplore、Elsevier/ScienceDirect、SpringerLink、ASME Digital Collection、Nature 系列、MDPI（仅取高引用/高质量）、Crossref/DOI 注册信息。 |
| `triage` | `thu_liqi_phd_thesis` | `Sensors` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/模型驱动的故障诊断方法/references (1).bib:89` | journal   = {Sensors}, |
| `triage` | `thu_liqi_phd_thesis` | `Applied Sciences` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/模型驱动的故障诊断方法/references (1).bib:109` | journal   = {Applied Sciences}, |
| `triage` | `thu_liqi_phd_thesis` | `Sensors` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:8` | journal      = {Sensors}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:14` | publisher    = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:15` | url          = {https://www.mdpi.com/1424-8220/25/9/2952}, |
| `triage` | `thu_liqi_phd_thesis` | `Applied Sciences` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:22` | journal      = {Applied Sciences}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:28` | publisher    = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:29` | url          = {https://www.mdpi.com/2076-3417/14/2/898}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:36` | journal      = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:91` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:92` | url          = {https://www.mdpi.com/2227-9717/8/9/1123}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:94` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:116` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:117` | url          = {https://www.mdpi.com/1996-1944/18/2/324}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:119` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:174` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:175` | url          = {https://www.mdpi.com/2073-8994/9/5/69}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:177` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:198` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:199` | url          = {https://www.mdpi.com/1099-4300/25/3/442}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:201` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:206` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:207` | url          = {https://www.mdpi.com/2227-9717/13/1/48}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:209` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:237` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:238` | url          = {https://www.mdpi.com/2227-7390/8/8/1308}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:240` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:253` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:254` | url          = {https://www.mdpi.com/1424-8220/24/18/6021}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:256` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:342` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:343` | url          = {https://www.mdpi.com/1996-1073/13/2/310}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:345` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:369` | publisher    = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:370` | url          = {https://www.mdpi.com/1996-1073/15/15/5703}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:441` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:442` | url          = {https://www.mdpi.com/1424-8220/25/13/3912}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:444` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:449` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:450` | url          = {https://www.mdpi.com/2071-1050/16/23/10651}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:452` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:491` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:492` | url          = {https://www.mdpi.com/2226-4310/11/9/773}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:494` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:529` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:530` | url          = {https://www.mdpi.com/2076-3417/10/15/5298}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.bib:532` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:651` | MDPI, 访问时间为 一月 6, 2026， |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:652` | \url{https://www.mdpi.com/1424-8220/25/9/2952}\\ |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:655` | Components, Trustworthiness, and Future Trends - MDPI, 访问时间为 一月 |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:656` | 6, 2026， \url{https://www.mdpi.com/2076-3417/14/2/898}\\ |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:683` | Processes - MDPI, 访问时间为 一月 6, 2026， |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:684` | \url{https://www.mdpi.com/2227-9717/8/9/1123}\\ |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:698` | Manufacturing - MDPI, 访问时间为 一月 6, 2026， |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:699` | \url{https://www.mdpi.com/1996-1944/18/2/324}\\ |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:724` | Information - MDPI, 访问时间为 一月 6, 2026， |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:725` | \url{https://www.mdpi.com/2073-8994/9/5/69}\\ |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:736` | Adaptive Nonlinear Membership Function - MDPI, 访问时间为 一月 6, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:737` | 2026， \url{https://www.mdpi.com/1099-4300/25/3/442}\\ |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:740` | on Artificial Intelligence and Signal Processing: A Review - MDPI, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:742` | \url{https://www.mdpi.com/2227-9717/13/1/48}\\ |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:758` | Hydraulic Engineering Investment Decisions - MDPI, 访问时间为 一月 6, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:759` | 2026， \url{https://www.mdpi.com/2227-7390/8/8/1308}\\ |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:766` | Reliability for Embedded Systems Based on SysML Models - MDPI, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:768` | \url{https://www.mdpi.com/1424-8220/24/18/6021}\\ |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:808` | Using Population-Based Incremental Learning - MDPI, 访问时间为 一月 6, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:809` | 2026， \url{https://www.mdpi.com/1996-1073/13/2/310}\\ |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:815` | Fault Diagnosis Using Bond Graphs in an Expert System - MDPI, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:817` | \url{https://www.mdpi.com/1996-1073/15/15/5703}\\ |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:855` | \url{https://www.mdpi.com/1424-8220/25/13/3912}\\ |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:858` | Enhancing Reliability in Sustainable Systems - MDPI, 访问时间为 一月 |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:859` | 6, 2026， \url{https://www.mdpi.com/2071-1050/16/23/10651}\\ |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:880` | Diagnosis Based on Logic Diagram Model - MDPI, 访问时间为 一月 6, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:881` | 2026， \url{https://www.mdpi.com/2226-4310/11/9/773}\\ |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:899` | An Ontological Metro Accident Case Retrieval Using CBR and NLP - MDPI, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/1-引言/知识驱动符号驱动故障方法/knowledge_sym_method_research.tex:901` | \url{https://www.mdpi.com/2076-3417/10/15/5298}\\ |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/理论部分/2、神经符号网络学习技术/neuro_symbolic_refs (1).bib:45` | publisher = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/理论部分/2、神经符号网络学习技术/neuro_symbolic_refs (1).bib:46` | url       = {https://www.mdpi.com/2226-4310/10/7/644}, |
| `triage` | `thu_liqi_phd_thesis` | `Electronics` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/理论部分/2、神经符号网络学习技术/neuro_symbolic_refs (1).bib:151` | journal = {Electronics}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/理论部分/2、神经符号网络学习技术/neuro_symbolic_refs (1).bib:157` | url     = {https://www.mdpi.com/2079-9292/14/24/4809}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/理论部分/2、神经符号网络学习技术/neuro_symbolic_refs (1).bib:373` | title   = {Human-in-the-Loop XAI for Predictive Maintenance: A Systematic Review of Interactive Systems and Their Effectiveness in Maintenance Decision-Making - MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/理论部分/2、神经符号网络学习技术/neuro_symbolic_refs (1).bib:374` | url     = {https://www.mdpi.com/2079-9292/14/17/3384}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/理论部分/2、神经符号网络学习技术/neuro_symbolic_refs (1).bib:403` | title   = {Understanding Physics-Informed Neural Networks: Techniques, Applications, Trends, and Challenges - MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/plan/reference/理论部分/2、神经符号网络学习技术/neuro_symbolic_refs (1).bib:404` | url     = {https://www.mdpi.com/2673-2688/5/3/74}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:185` | url     = {https://www.mdpi.com/2075-1702/11/7/692}, |
| `triage` | `thu_liqi_phd_thesis` | `Applied Sciences` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:254` | journal = {Applied Sciences}, |
| `triage` | `thu_liqi_phd_thesis` | `Scientific Reports` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:276` | journal = {Scientific Reports}, |
| `triage` | `thu_liqi_phd_thesis` | `Sensors` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:517` | journal = {Sensors}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:894` | journal = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:935` | url     = {https://www.mdpi.com/2075-1702/11/5/519}, |
| `triage` | `thu_liqi_phd_thesis` | `Sensors` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:1032` | journal = {Sensors}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:1077` | url     = {https://www.mdpi.com/2077-1312/12/12/2284} |
| `triage` | `thu_liqi_phd_thesis` | `Sensors` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:1083` | journal = {Sensors}, |
| `triage` | `thu_liqi_phd_thesis` | `Sensors` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:1096` | journal = {Sensors}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:1097` | url     = {https://www.mdpi.com/1424-8220/23/8/4007}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:1428` | journal = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `Applied Sciences` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:1437` | journal = {Applied Sciences}, |
| `triage` | `thu_liqi_phd_thesis` | `Applied Sciences` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:1446` | journal = {Applied Sciences}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:1517` | url     = {https://www.mdpi.com/1099-4300/21/11/1089}, |
| `triage` | `thu_liqi_phd_thesis` | `Sensors` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:1556` | journal = {Sensors}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:1557` | url     = {https://www.mdpi.com/1424-8220/22/6/2192}, |
| `triage` | `thu_liqi_phd_thesis` | `Sensors` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:1837` | journal = {Sensors}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:1843` | url     = {https://www.mdpi.com/1424-8220/19/7/1693} |
| `triage` | `thu_liqi_phd_thesis` | `Sensors` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:1850` | journal = {Sensors}, |
| `triage` | `thu_liqi_phd_thesis` | `Sensors` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:2076` | journal = {Sensors}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:2101` | journal = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:2122` | journal = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `Electronics` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:2187` | journal = {Electronics}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:3263` | journal = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:3280` | journal = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:3449` | journal = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:3490` | journal = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:3508` | journal = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:3533` | journal = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:3541` | journal = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:3594` | journal = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:3732` | journal = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `Sensors` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4124` | journal   = {Sensors}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4130` | publisher = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4131` | url       = {https://www.mdpi.com/1424-8220/25/9/2952}, |
| `triage` | `thu_liqi_phd_thesis` | `Applied Sciences` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4138` | journal   = {Applied Sciences}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4144` | publisher = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4145` | url       = {https://www.mdpi.com/2076-3417/14/2/898}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4152` | journal   = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4207` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4208` | url          = {https://www.mdpi.com/2227-9717/8/9/1123}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4210` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4232` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4233` | url          = {https://www.mdpi.com/1996-1944/18/2/324}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4235` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4290` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4291` | url          = {https://www.mdpi.com/2073-8994/9/5/69}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4293` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4314` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4315` | url          = {https://www.mdpi.com/1099-4300/25/3/442}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4317` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4322` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4323` | url          = {https://www.mdpi.com/2227-9717/13/1/48}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4325` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4353` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4354` | url          = {https://www.mdpi.com/2227-7390/8/8/1308}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4356` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4369` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4370` | url          = {https://www.mdpi.com/1424-8220/24/18/6021}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4372` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4458` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4459` | url          = {https://www.mdpi.com/1996-1073/13/2/310}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4461` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4485` | publisher = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4486` | url       = {https://www.mdpi.com/1996-1073/15/15/5703}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4557` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4558` | url          = {https://www.mdpi.com/1424-8220/25/13/3912}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4560` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4565` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4566` | url          = {https://www.mdpi.com/2071-1050/16/23/10651}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4568` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4607` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4608` | url          = {https://www.mdpi.com/2226-4310/11/9/773}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4610` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4645` | organization = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4646` | url          = {https://www.mdpi.com/2076-3417/10/15/5298}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4648` | note         = {TODO:VERIFY authors/year/DOI from MDPI citation section} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4881` | title = {Fault Diagnosis for Complex Equipment Based on Belief Rule Base with Adaptive Nonlinear Membership Function - MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4882` | url   = {https://www.mdpi.com/1099-4300/25/3/442} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4886` | title = {Dynamic Fault Tree Generation and Quantitative Analysis of System Reliability for Embedded Systems Based on SysML Models - MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4887` | url   = {https://www.mdpi.com/1424-8220/24/18/6021} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4906` | title = {Knowledge Graph Construction Method for Commercial Aircraft Fault Diagnosis Based on Logic Diagram Model - MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4907` | url   = {https://www.mdpi.com/2226-4310/11/9/773} |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4960` | journal   = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:4963` | note      = {Conference Name: IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:5024` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:5709` | url        = {https://www.mdpi.com/2076-3417/10/20/7302}, |
| `triage` | `thu_liqi_phd_thesis` | `Applied Sciences` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:5714` | journal    = {Applied Sciences}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:6220` | url       = {https://www.mdpi.com/2813-0324/2/1/19}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:6225` | publisher = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:6526` | journal    = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:7608` | journal    = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:8459` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:8462` | note     = {Conference Name: IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:9298` | journal    = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:9318` | journal   = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:9475` | journal   = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:9841` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:10212` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:10902` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:10905` | note     = {Conference Name: IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:11393` | url        = {https://www.mdpi.com/2504-4494/6/1/10}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:11468` | url        = {https://www.mdpi.com/2504-4494/6/1/10}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:11782` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:11786` | note     = {Conference Name: IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:13702` | journal    = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:13705` | note       = {Conference Name: IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:15755` | journal    = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:15758` | note       = {Conference Name: IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:16131` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:16134` | note     = {Conference Name: IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:17217` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:17220` | note     = {Conference Name: IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:18900` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:18903` | note     = {Conference Name: IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:19263` | journal   = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:19266` | note      = {Conference Name: IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:19287` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:19290` | note     = {Conference Name: IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:19451` | journal   = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:19454` | note      = {Conference Name: IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:19598` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:20183` | journal   = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:20186` | note      = {Conference Name: IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:20863` | url      = {http://www.mdpi.com/2076-3417/8/10/1786}, |
| `triage` | `thu_liqi_phd_thesis` | `Applied Sciences` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:20868` | journal  = {Applied Sciences}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:21402` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:21418` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:21434` | journal    = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:21449` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:21464` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:21478` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:21493` | journal    = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:21508` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:21523` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:21537` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:21551` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:21565` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:21578` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:21592` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:21606` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:21620` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:21635` | journal    = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:23532` | journal  = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:24476` | journal  = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:24756` | journal  = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `Scientific Reports` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:25308` | journal  = {Scientific Reports}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:25319` | journal  = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `Scientific Reports` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:25388` | journal  = {Scientific Reports}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:25462` | journal  = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:25482` | journal  = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:25891` | journal = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:26116` | note       = {Publisher: MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:26128` | note       = {Publisher: MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Access` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:26136` | journal    = {IEEE Access}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:26211` | journal  = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:26283` | journal    = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:26845` | journal   = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:26863` | journal   = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `IEEE Transactions on Instrumentation and Measurement` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:27488` | journal    = {IEEE Transactions on Instrumentation and Measurement}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:27758` | publisher = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:27759` | url       = {https://www.mdpi.com/2226-4310/10/7/644}, |
| `triage` | `thu_liqi_phd_thesis` | `Applied Sciences` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:28019` | journal   = {Applied Sciences}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:28025` | publisher = {MDPI} |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:28044` | publisher = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `MDPI` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:28079` | publisher = {MDPI}, |
| `triage` | `thu_liqi_phd_thesis` | `Sensors` | `paper/UXFD_paper/thu_liqi_phd_thesis/ref/refs.bib:28119` | journal = {Sensors}, |

## Blockers
