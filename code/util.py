def save_xml(boxes, path):
    filename = path.split("/")[-1]
    width = 2560
    height = 1440
    object_template = lambda label, rect: f"""<object>
        <name>{label}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{int((rect[0]-rect[2]/2)*width)}</xmin>
            <ymin>{int((rect[1]-rect[3]/2)*height)}</ymin>
            <xmax>{int((rect[0]+rect[2]/2)*width)}</xmax>
            <ymax>{int((rect[1]+rect[3]/2)*height)}</ymax>
        </bndbox>
    </object>"""
    objects = ""
    for box in boxes:
        label, confidence, rect = box    
        objects += object_template(label, rect) + "\n"
                
    output = f"""<annotation>
        <folder>images</folder>
        <filename>{filename}</filename>
        <path>{path}</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>{width}</width>
            <height>{height}</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
        {objects}
    </annotation>"""

    with open(path.replace(".jpg", ".xml"), "w") as f:
        f.write(output)
