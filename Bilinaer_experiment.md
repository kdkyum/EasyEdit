
First, run the collecting embedding at the bilinear folder:

```bash
python submit_collect_embeddings.py   --model_list "Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-2-7b-chat-hf,mistralai/Mistral-7B-Instruct-v0.3" --output_dir hidden_repr/person-city --input_csv_paths "datasets/counterfact_person-city_test.json","datasets/counterfact_person-city_train.json" --batch_size 8 --eval_batch_size 512
```

Next, run the bilinear probing experiment:

```bash
python scripts/slurm_batch_compare.py   --emb-dir hidden_repr/person-city   --train-dataset datasets/counterfact_person-city_train.json   --test-dataset datasets/counterfact_person-city_test.json   --relations city_country   --out-root outputs/cli_batch_person-city   --device cpu   --num-workers 4 --submit-slurm --time 12:00:00 --lambda-R 1e-3
```

<!-- python3 scripts/slurm_batch_compare.py --emb-dir hidden_repr/person-city --train-dataset datasets/counterfact_person-city_train_wikidata.json --test-dataset datasets/counterfact_person-city_test_wikidata.json --relations city_country --out-root outputs/cli_batch_person-city_wikidata --aggregate-only -->

then, plot the results:

```bash
python scripts/slurm_batch_compare.py --emb-dir hidden_repr/person-city --train-dataset datasets/counterfact_person-city_train.json --test-dataset datasets/counterfact_person-city_test.json --relations all --out-root outputs/cli_batch_person-city --aggregate-only
```

Now, run the editing experiment:

```bash
sbatch scripts/instruct_models/run_llama2_ft_person_city.sh
sbatch scripts/instruct_models/run_mistral7b_ft_person_city.sh
sbatch scripts/instruct_models/run_qwen2.5_7b_ft_person_city.sh
```

editing result plot:

```bash
python scripts/plot_results.py --results_dir results --out_dir results/plots
```



# TODO:

1. correct_indices.npz를 활용해서, 오직 lm이 알고 있는 fact에 대해서만 결과를 취합해서 비교하기 (run_editing.py 그리고 evaluation 파일들 바꿔서 portability chat template으로 평가하기 + 2-hop에 대해서 실제로 생성한 prompt로 결과로 저장하도록 변경.)
2. bilinear로 맞춘것 bilinear_correct = 1 샘플이 실제 editing 한 샘플에서도 locality와 portability (2-hop reasoning) 가 좋은 성능을 보여주는지?