
First, run the collecting embedding at the bilinear folder:

```bash
python submit_collect_embeddings.py   --model_list "Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-2-7b-chat-hf,mistralai/Mistral-7B-Instruct-v0.3" --output_dir hidden_repr/person-city_wikipedia  --input_csv_paths "datasets/counterfact_person-city_test_wikipedia.json","datasets/counterfact_person-city_train_wikipedia.json" --batch_size 1 --eval_batch_size 512
```

Next, run the bilinear probing experiment:

```bash
for lr in 1 0.99999 0.9999 0.999 0.99 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do python scripts/slurm_batch_compare.py --emb-dir hidden_repr/person-city_wikipedia --train-dataset datasets/counterfact_person-city_train_wikipedia.json --test-dataset datasets/counterfact_person-city_test_wikipedia.json --relation person-city --out-root "outputs/truncated_person-city_wikipedia2/variance_${lr}" --device cpu --num-workers 16 --submit-slurm --time 12:00:00 --lambda-R 0 --variance-threshold ${lr}; done
```

<!-- python3 scripts/slurm_batch_compare.py --emb-dir hidden_repr/person-city --train-dataset datasets/counterfact_person-city_train_wikidata.json --test-dataset datasets/counterfact_person-city_test_wikidata.json --relations city_country --out-root outputs/cli_batch_person-city_wikidata --aggregate-only -->

then, plot the results:

```bash
python scripts/plot_rank_vs_accuracy.py outputs/truncated_person-city_wikipedia 
```

Now, run the editing experiment:

```bash
sbatch scripts/instruct_models/run_llama2_ft_person_city_wikipedia.sh; sbatch scripts/instruct_models/run_mistral7b_ft_person_city_wikipedia.sh; sbatch scripts/instruct_models/run_qwen2.5_7b_ft_person_city_wikipedia.sh; sbatch scripts/instruct_models/run_llama3_ft_person_city_wikipedia.sh
```

editing result plot:

```bash
python plot_results.py 
```

TODO: bilinear로 맞춘것 bilinear_correct = 1 샘플이 실제 editing 한 샘플에서도 locality와 portability (2-hop reasoning) 가 좋은 성능을 보여주는지?