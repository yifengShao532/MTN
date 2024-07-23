# MTN

Source code for "Multi-view Transformer-based Network for Prerequisite Learning in Concept Graphs".

## Environment

```bash
pip install -r requirements.txt
```

## Data Processing & Concept Graph Construction

```bash
python concept_graph_construction.py
```

## Usage

Download the benchmarks from [here](https://drive.google.com/file/d/1A-_FpBxgtuYurP7DRluBD39h0LCOw4Ai/view?usp=sharing) and extract them to the `./` directory.

Run the following command to train the model on different datasets and settings.
    
For conventional setting:

```bash
python main.py --data_path benchmarks/conventional_setting/LectureBank/ --fold_id 0
```

For hard setting:

```bash
python main.py --data_path benchmarks/hard_setting/LectureBank/ --fold_id 0
```
