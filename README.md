# Transformer Model from Scratch

![transformer.png](https://github.com/ultimateabhi719/transformer_from_scratch/blob/24927be0521f4d741c1f1b5ad9bde7101c20e787/transformer.png)

The repository implements the transformer model for translation using: 
- greedy decoding 
- NLL loss criterion

## Data Training:
1. Install the pytorch\_transformer package with the following command:
`pip install -e .`
2. Change the model parameters, dataset & tokenizer parameters, and the training parameters in the file `scripts/run.py` if needed. `data_params['dataset_path']`, `data_params['tokenizers']`, `data_params['lang']` correspond to the hugging face dataset parameters. Use the `exploratory-data-analysis.ipynb` notebook to save text data in hugging face dataset format.
3. Run scripts/run.py with the `pt.optimize\_optimizer..` line uncommented and `pt.main..` line commented to find out the optimal learning rate and optimizer to use (uses optuna).
4. Set the lr in train\_params and the optimizer in `train\_model@main.py`
5. Uncomment back the `pt.main` line and comment the `pt.optimizer\_optimizer` line. Now run `scripts/run.py`

## Evaluate/Predict Model
1. Once training is complete the model is saved in the train\_params['save\_prefix'] directory
2. Run the following command to evaluate the trained model:
`python scripts/eval.py <saved-model-directory/path> -i "<input-text-in-source-language>"`

## Deploy Model
1. Run `scripts/bentoml/save\_model.py <saved-model-dir> <bento-model-name>` to save the trained model to bentoml. Note: You can see the saved bentoml models using : `bentoml models list` 
2. run the service.py script using bento with the following command to serve the model on port 3000:
`bentoml serve -p 3000 --api-workers 1 --working-dir scripts/bentoml service:svc`



## Further Improvements:
- variable batch size merging groups of similar length together to increase training speed
- Take loss = bleu score instead of NLL. To improve performance
- increase model size
- identify better dataset 
