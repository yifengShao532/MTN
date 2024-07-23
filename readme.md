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

Follow the steps below to use the MTN for training on various datasets and settings.

1. **Download the Benchmarks**

   Download the benchmarks from [this link](https://drive.google.com/file/d/1ufYxZG4HPIAMzW1bxeGTnrEM2w4xQOcn/view?usp=sharing) and extract them to the `./` directory.

2. **Train the Model**

   To train the model on different datasets and settings, use the following commands:

   #### Conventional Setting

   Taking the LectureBank dataset fold 0 as an example, run the following command:

    ```bash
    python main.py --data_path benchmarks/conventional_setting/LectureBank/ --fold_id 0
    ```

   #### Hard Setting

    Similarly, for hard setting:

    ```bash
    python main.py --data_path benchmarks/hard_setting/LectureBank/ --fold_id 0
    ```
