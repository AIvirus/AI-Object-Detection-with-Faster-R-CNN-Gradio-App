
# AI Object Detection App 🚀

Detect objects in images using a **pre-trained Faster R-CNN model** and visualize the results with colorful bounding boxes.  
Built with **PyTorch**, **Gradio**, and **PIL**!

## ✨ Features
- Upload any image and detect multiple objects.
- Each object has a **different colored bounding box**.
- Adjust the **confidence threshold** with a slider.
- Live demo with **Gradio's public share link**.

## 🖼 Demo
| Upload an Image | Detect Objects |
|:---------------:|:--------------:|
| ![Upload](https://via.placeholder.com/200x150?text=Upload+Image) | ![Result](https://via.placeholder.com/200x150?text=Detection+Result) |

## 📦 Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision gradio pillow
```

## 🛠 Usage

```bash
python app.py
```
(Replace `app.py` with your filename if it's different.)

- A Gradio interface will launch.
- Upload an image.
- Adjust the confidence threshold if needed.
- See detected objects highlighted with colorful bounding boxes.

## 🧠 Model Details
- **Model**: Faster R-CNN with ResNet-50 FPN backbone.
- **Trained on**: COCO dataset (80 common objects).

## 📁 Project Structure

```
├── app.py          # Main Python script
├── requirements.txt
└── README.md       # Project documentation
```

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first.

## 📜 License
[MIT License](LICENSE)
