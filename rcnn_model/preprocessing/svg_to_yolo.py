import xml.etree.ElementTree as ET
import re
import os
import glob

namespace = {"svg": "http://www.w3.org/2000/svg"}


YOLO_CLASSES = {
    "Door": 0,
    "Window": 1,
    "Space": 2
}

def extract_svg_elements(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    svg_width = float(root.get("width", "1"))
    svg_height = float(root.get("height", "1"))

    # floorplans = {}
    floorplans = {"Door": [], "Window": [], "Space": []}
    for floorplan in root.findall(".//svg:g[@class]", namespaces=namespace):
        class_attr = floorplan.get("class", "").strip()

        
        if class_attr in ["Floorplan Floor-1", "Floorplan Floor-2"]:

            for element in floorplan.iter():
                class_attr = element.get("class")
                if class_attr:
                    if any(cat in class_attr for cat in YOLO_CLASSES.keys()):
                        polygons = []
                        for poly in element.findall(".//svg:polygon", namespaces=namespace):
                            points = poly.get("points")
                            if points:
                                polygons.append(points)

                        if polygons:
                            category = next((cat for cat in YOLO_CLASSES if cat in class_attr), None)
                            print(type(polygons[0]))
                            if category:
                                bbox = get_bounding_box(polygons[0], svg_width, svg_height)

                                if "Space" in class_attr:
                                    name_label = re.sub(r'\b[Ss]pace\b', '', class_attr).strip()
                                    floorplans["Space"].append({
                                        "name": name_label, "bbox": bbox
                                    })
                                else:
                                    floorplans[category].append({"bbox": bbox})

    return floorplans, svg_width, svg_height

def get_bounding_box(polygons, svg_width, svg_height):
    """Compute YOLO bounding box from polygon points."""
    all_x, all_y = [], []
    print(polygons)
    # for polygon in polygons:
        # print(polygon)
    points = polygons.strip().split(" ")
    for point in points:
        x, y = map(float, point.split(","))
        all_x.append(x)
        all_y.append(y)
    print(all_x, all_y)
    # Bounding Box Calculation
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Convert to YOLO format (normalized)
    x_center = (x_min + x_max) / 2 / svg_width
    y_center = (y_min + y_max) / 2 / svg_height
    width = (x_max - x_min) / svg_width
    height = (y_max - y_min) / svg_height

    return (x_center, y_center, width, height)

def save_yolo_annotations(floorplans, output_dir, filename):
    """Save extracted bounding boxes in YOLO format."""
    os.makedirs("dataset", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

   
    output_file = f"{output_dir}/{filename}.txt"

    with open(output_file, "w") as f:
        for category, elements in floorplans.items():
            class_id = YOLO_CLASSES[category]

            for element in elements:
                bbox = element["bbox"]
                yolo_line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                f.write(yolo_line)

    print(f"YOLO annotations saved in '{output_dir}'")


input_folder = "../cubicasa5k/high_quality/"  
output_folder = "dataset/yolo_annotations" 

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print(f"Fixed Working Directory: {os.getcwd()}")

subfolders = glob.glob(os.path.join(input_folder, "*"))  

for subfolder in subfolders:
    svg_file = os.path.join(subfolder, "model.svg")  
    
    if os.path.exists(svg_file):  
        filename = os.path.basename(subfolder)  
        print(f"Processing: {svg_file} ...")
        
        floorplans, svg_width, svg_height = extract_svg_elements(svg_file)
        save_yolo_annotations(floorplans, output_folder, filename)

print(" All SVG files have been processed!")
