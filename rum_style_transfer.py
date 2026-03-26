from utils import load_img_from_url, deprocess_img
from style import get_model, run_style_transfer
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------
# Image URLs
# ----------------------------
content_url = "https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d?auto=format&fit=crop&w=500&q=80"
style_url   = "https://images.unsplash.com/photo-1526948128573-703ee1aeb6fa?auto=format&fit=crop&w=500&q=80"

# Load images
content_img = load_img_from_url(content_url)
style_img = load_img_from_url(style_url)

# ----------------------------
# Run Style Transfer
# ----------------------------
model, _, _ = get_model()

stylized_image = run_style_transfer(
    model,
    content_img,
    style_img,
    iterations=300
)

# ----------------------------
# Convert Images
# ----------------------------
content_display = deprocess_img(content_img)
style_display = deprocess_img(style_img)
stylized_display = deprocess_img(stylized_image)

# Save output
Image.fromarray(stylized_display).save("stylized_output.jpg")

# ----------------------------
# Display Images
# ----------------------------
plt.figure(figsize=(12, 4))

# Content
plt.subplot(1, 3, 1)
plt.imshow(content_display)
plt.title("Content Image")
plt.axis('off')

# Style
plt.subplot(1, 3, 2)
plt.imshow(style_display)
plt.title("Style Image")
plt.axis('off')

# Output
plt.subplot(1, 3, 3)
plt.imshow(stylized_display)
plt.title("Stylized Image")
plt.axis('off')

plt.tight_layout()
plt.show()