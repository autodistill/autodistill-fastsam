**⚠️ This is an example README for use in creating a Base Model. You will need to adjust this document for the model you are using.**

<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill FastSAM Module

This repository contains the code supporting the FastSAM base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) is a segmentation model trained on 2% of the SA-1B dataset used to train the [Segment Anything Model](https://github.com/facebookresearch/segment-anything).

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [FastSAM Autodistill documentation](https://autodistill.github.io/autodistill/base_models/fastsam/).

## Installation

To use FastSAM with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-fastsam
```

## Quickstart

```python
from autodistill_fastsam import FastSAM

# define an ontology to map class names to our FastSAM prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = FastSAM(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)
base_model.label("./context_images", extension=".jpeg")
```


## License

This project is licensed under an [Apache 2.0 license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!