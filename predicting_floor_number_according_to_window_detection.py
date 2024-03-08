import cv2
import numpy as np

from google.colab import files
from google.colab.patches import cv2_imshow
from sklearn.cluster import DBSCAN


# Upload image
uploaded = files.upload()
image_data = list(uploaded.values())[0]


# Resmi yükleme
image = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Smooth the image using a Gaussian filter
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Apply morphological operations to clean up the edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Find contours in the closed image
contours, hierarchy = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Define parameters for the rectangle filtering process
min_area = 200
max_area = 2500
min_width, min_height = 10, 10
max_width, max_height = 400, 400
min_ratio, max_ratio = 0.72, 1.12

# Iterate through all contours found and filter rectangles based on the defined parameters
rectangles = []
for contour in contours:
    area = cv2.contourArea(contour)
    x, y, width, height = cv2.boundingRect(contour)
    ratio = width / height
    # if min_ratio < ratio < max_ratio:
    #     rectangles.append((x, y, width, height))

    # if min_ratio < ratio < max_ratio and min_width < width < max_width and min_height < height < max_height:
    #     rectangles.append((x, y, width, height))

    if min_area < area < max_area:
      ratio = width / height
      if min_ratio < ratio < max_ratio and min_width < width < max_width and min_height < height < max_height:
        rectangles.append((x, y, width, height))



# Iterate through all rectangles and draw them onto the original image
for rect in rectangles:
    cv2.rectangle(image, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)




eps = 40  # İki dikdörtgenin y değerleri arasındaki maksimum fark
min_samples = 1  # Minimum örnekleme sayısı

# Dikdörtgenlerin y değerlerini kullanarak kümeleme işlemini gerçekleştirin
Y = np.array([rect[1] for rect in rectangles]).reshape(-1, 1)
dbscan_y = DBSCAN(eps=eps, min_samples=min_samples).fit(Y)

# Küme etiketlerini alın
cluster_labels = dbscan_y.labels_

# Her bir küme için dikdörtgenleri gruplandırın
rect_clusters = {}
for i, rect in enumerate(rectangles):
    cluster_label = cluster_labels[i]
    if cluster_label in rect_clusters:
        rect_clusters[cluster_label].append(rect)
    else:
        rect_clusters[cluster_label] = [rect]

# Küme sayısını yazdırmak için metin oluşturun
n_clusters = len(rect_clusters)  # Küme sayısı
text = f"Floor Count: {n_clusters}"

font = cv2.FONT_HERSHEY_SIMPLEX
text_position = (50, 100)  # Metnin konumu
text_color = (0, 0, 255)  # Metin rengi
text_thickness = 1  # Metin kalınlığı
text_line_type = cv2.LINE_AA  # Metin satır tipi
font_scale = 0.8  # Font büyüklüğü ölçeği
cv2.putText(image, text, text_position, font, font_scale, text_color, text_thickness, text_line_type)






# Display the final image with rectangles drawn around the detected windows
cv2_imshow(image)
cv2.waitKey(0)
cv2.destroyAllWindows()