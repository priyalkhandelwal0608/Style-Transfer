#  Neural Style Transfer using TensorFlow

Transform ordinary images into artistic masterpieces using **Neural Style Transfer (NST)** powered by **VGG19** and **TensorFlow**.

---

##  Project Overview

This project implements Neural Style Transfer by combining:

* **Content Image** → Structure of the image
* **Style Image** → Artistic texture & patterns

The result is a **stylized image** that blends both.

---

##  How It Works

* Uses **VGG19 pretrained model**
* Extracts:

  * **Content features** (deep layers)
  * **Style features** (Gram matrix of multiple layers)
* Optimizes a generated image using:

  * Content Loss
  * Style Loss
* Uses **Gradient Descent (Adam optimizer)**

---

##  Project Structure

```
Style-Transfer/
│── style.py                 # Core NST logic (model + loss + training)
│── utils.py                 # Image processing + Gram matrix
│── run_style_transfer.py    # Main script to run NST
│── stylized_output.jpg      # Output image (generated)
│── README.md                # Project documentation
```

---

##  Installation

```bash
pip install tensorflow pillow matplotlib requests
```

---

##  Run the Project

```bash
python run_style_transfer.py
```

---

##  Output

The script will:

* Display:

  * Content Image
  * Style Image
  * Stylized Output
* Save output as:

```
stylized_output.jpg
```

---

##  Parameters (Tunable)

Inside `style.py`:

```python
iterations = 300        # Increase for better quality
content_weight = 1e4   # Preserve structure
style_weight = 1e-2    # Artistic effect
```

---

##  Features

 Uses pretrained VGG19
 Custom loss functions
 High-quality stylization
 Works with any image URL
 Clean modular code


