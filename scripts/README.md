# Running a recognizer over recordings

## Local recordings

### Step 1. Generate Embeddings

The recognizer requires embeddings to be generated. 

Go to https://github.com/QutEcoacoustics/perch-runner/tree/main/scripts and follow steps 1, 2 and 3. 


### Step 2. Download script

If you followed the instructions on perch-runner scripts to embed audio, you will have embeddings and have docker installed. 

Right click on the link and choose "save link as"
<a href="https://raw.githubusercontent.com/QutEcoacoustics/embeddings-classifier/main/scripts/run_container.ps1" download>run_container.ps1</a>


### Step 3. Run the recognizer

1. Open a terminal window
2. Change directory to the directory where your downloaded run_container.ps1 script is
3. Run the following command:
   `powershell -ExecutionPolicy Bypass -File .\run_container.ps1 -input [path_to_embeddings_folder] -OutputFolder [path_to_classification_output_folder] -ConfigFile [path_to_classifier_json]`


Notes
- In the command above, replace the placeholders with your real embeddings and output folder and classifier json file. The output folder is where the classifications files will get saved.

## Combine recognizer config files

To build one classifier config from multiple individual recognizer JSON files, create a manifest:

```json
{
   "recognizer_paths": [
      "/path/to/recognizer_one.json",
      "/path/to/recognizer_two.json"
   ]
}
```

Then run:

```bash
python scripts/create_config_list.py \
   --manifest /path/to/manifest.json \
   --output /path/to/config_list.json
```

Use the generated `config_list.json` as the `config` value in the run parameters passed to `run_model_on_ecosounds_2.py`.