# Plant-Sense

**UNIBO Multimedia Data Management Project**

This project focuses on **image retrieval** for plant data, specifically analyzing image similarity based on color histograms and SIFT features.

---

## 📦 Installation

Before running the scripts, install the required dependencies:

```bash
pip install opencv-python numpy matplotlib pandas scikit-learn python-dotenv
```

## ⚙️ Environment Setup

You must create a `.env` file in the root of your project with the following variables:

```env
IMG_FOLDER=path/to/image_collection
IMG_PLOT_FOLDER=path/to/output_plots
QUERY_IMAGE=path/to/query_image.jpg
```

**Explanation:**

- `IMG_FOLDER`: Folder containing the plant images to compare.
- `IMG_PLOT_FOLDER`: Folder where output histogram and SIFT keypoint images will be saved.
- `QUERY_IMAGE`: Path to the image used as a query for retrieval.

---

## 🚀 How to Run

After setting up the `.env` file, execute the image retrieval script:

```bash
python image_retrieval.py
```

The script will:

- Load the query image and compute color histogram and SIFT descriptors.
- Compare it to all images in the `IMG_FOLDER`.
- Save:
  - A histogram comparison plot for each image.
  - A SIFT keypoint visualization for each image.
  - Two ranking tables:
    - `result_hist.png`: similarity ranking based on color histogram.
    - `result_sift.png`: similarity ranking based on SIFT features.

---

## 📁 Project Structure

```
plant-sense/
├── image_retrieval.py
├── .env
├── result_hist.png
├── result_sift.png
├── img_plot/         # Output plots (generated automatically)
└── img/              # Your image collection
```

---

## 📝 Notes

- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- Ensure your OpenCV version includes SIFT (`opencv-python` already does).
- The project will automatically create the output folders if they don't exist.
