PYTHON ?= python
COMMON = --batch_size 96 --num_steps 300 --eval_every 20 --num_seeds 3
GROUPED = --group_size 8 --inner_epochs 4

.PHONY: smoke test figures sweep-influence sweep-staleness sweep-uncertainty sweep-token-credit sweep-self-distill sweep-entropy

smoke:
	$(PYTHON) -m compileall -q rl_sandbox
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method DG \
		--batch_size 16 --num_steps 2 --eval_every 1 --num_seeds 1 \
		--output /tmp/rl_sandbox_smoke.csv --verbose false

test:
	$(PYTHON) -m compileall -q rl_sandbox tests scripts
	$(PYTHON) -m unittest discover -s tests

figures:
	$(PYTHON) rl_sandbox/analysis/plot_evidence.py

sweep-influence:
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method DG $(COMMON) --output results/dg_token.csv
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method GRPO $(COMMON) $(GROUPED) --output results/grpo_token.csv
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method TPO $(COMMON) $(GROUPED) --output results/tpo_token.csv

sweep-staleness:
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method DG $(COMMON) --delay 4 --output results/replay_dg_delay4.csv
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method ReplayDG $(COMMON) --delay 4 --replay_capacity 5 --output results/replay_replaydg_cap5_delay4.csv
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method FreshDG $(COMMON) --delay 4 --replay_capacity 5 --output results/replay_freshdg_cap5_delay4.csv
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method FreshDG $(COMMON) --delay 4 --replay_capacity 32 --replay_age_decay 0.5 --output results/replay_freshdg_decay05_delay4.csv

sweep-uncertainty:
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method DG $(COMMON) --reward_noise 0.2 --reward_noise_mode false_positive_rare_token --output results/noise_dg_false_positive_rare.csv
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method UncertaintyDG $(COMMON) --reward_noise 0.2 --reward_noise_mode false_positive_rare_token --output results/noise_uncertaintydg_false_positive_rare.csv
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method RewardVarianceDG $(COMMON) --reward_noise 0.2 --reward_noise_mode false_positive_rare_token --output results/noise_rewardvariancedg_false_positive_rare.csv
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method ASPO $(COMMON) --reward_noise 0.2 --reward_noise_mode false_positive_rare_token --output results/noise_aspo_false_positive_rare.csv
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method R2VPO $(COMMON) --reward_noise 0.2 --reward_noise_mode false_positive_rare_token --output results/noise_r2vpo_false_positive_rare.csv

sweep-token-credit:
	$(PYTHON) -m rl_sandbox.train --task masked_reversal --method CE $(COMMON) --output results/masked_axis_ce.csv
	$(PYTHON) -m rl_sandbox.train --task masked_reversal --method DG $(COMMON) --output results/masked_axis_dg.csv
	$(PYTHON) -m rl_sandbox.train --task masked_reversal --method DGToken $(COMMON) --output results/masked_axis_dgtoken.csv
	$(PYTHON) -m rl_sandbox.train --task masked_reversal --method TPOToken $(COMMON) $(GROUPED) --output results/masked_axis_tpotoken.csv
	$(PYTHON) -m rl_sandbox.train --task masked_reversal --method GRPOToken $(COMMON) $(GROUPED) --output results/masked_axis_grpotoken.csv

sweep-self-distill:
	$(PYTHON) -m rl_sandbox.train --task chain_reversal --method CE --batch_size 96 --num_steps 1500 --eval_every 50 --num_seeds 3 --output results/chain_ce_1500.csv
	$(PYTHON) -m rl_sandbox.train --task chain_reversal --method SelfDistillDG --batch_size 96 --num_steps 1500 --eval_every 50 --num_seeds 3 --output results/chain_selfdistilldg_1500.csv
	$(PYTHON) -m rl_sandbox.train --task chain_reversal --method SCOPELite --batch_size 96 --num_steps 1500 --eval_every 50 --num_seeds 3 --output results/chain_scopelite_1500.csv

sweep-entropy:
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method DG $(COMMON) --entropy_diagnostics true --output results/entropy_dg.csv
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method DGEntropyGuard $(COMMON) --entropy_diagnostics true --output results/entropy_dgentropyguard.csv
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method ASPO $(COMMON) --entropy_diagnostics true --output results/entropy_aspo.csv
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method R2VPO $(COMMON) --entropy_diagnostics true --output results/entropy_r2vpo.csv
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method GRPO $(COMMON) $(GROUPED) --entropy_diagnostics true --output results/entropy_grpo.csv
	$(PYTHON) -m rl_sandbox.train --task token_reversal --method TPO $(COMMON) $(GROUPED) --entropy_diagnostics true --output results/entropy_tpo.csv
