# Step-7 正式健康版运行报告

## Status

FAILED

## Active Workflow

- `active_workflow`: workflow_0.1
- `active_stage`: Step-7
- `status`: step7-formal-run-in-progress

## Paths

- `step6_experiment_dir`: /Users/xuzijian/Desktop/竞赛/大数据竞赛/Experiment/workflow_0.1/experiments/exp_20260617_step6_workflow_0_1
- `experiment_dir`: /Users/xuzijian/Desktop/竞赛/大数据竞赛/Experiment/workflow_0.1/experiments/exp_20260617_step7_local_score_workflow_0_1
- `output_dir`: /Users/xuzijian/Desktop/竞赛/大数据竞赛/Experiment/workflow_0.1/experiments/exp_20260617_step7_local_score_workflow_0_1/outputs/step7
- `official_scoring_workspace`: /Users/xuzijian/Desktop/竞赛/大数据竞赛/Experiment/workflow_0.1/experiments/exp_20260617_step7_local_score_workflow_0_1/official_scoring_workspace

## Params

- mode: local-score
- official_script_path: /Users/xuzijian/Desktop/竞赛/大数据竞赛/THU-BDC2026-main/test/score_self.py
- team_name: team_name
- test_data_path: /Users/xuzijian/Desktop/竞赛/大数据竞赛/THU-BDC2026-main/data/test.csv

## Input Metrics

- input_candidate_date: 2026-06-15
- input_result_codes: 5 codes
- input_selected_count: 5
- input_step6_experiment: exp_20260617_step6_workflow_0_1
- input_step6_result_path: /Users/xuzijian/Desktop/竞赛/大数据竞赛/Experiment/workflow_0.1/experiments/exp_20260617_step6_workflow_0_1/outputs/step6/step6_result.csv
- input_total_weight: 1.0

## Output Metrics

_not available_

## Error

stock_contribution must cover exactly all frozen_result stocks in local-score mode; local-score result_status must be SCORE_SUCCESS; local-score final_score must be numeric; step7_leakage_check.csv has non-PASS rows: ['test_data_is_future_of_candidate_date', 'official_script_completed', 'final_score_not_negative_999', 'stock_contribution_matches_final_score']
