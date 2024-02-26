from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
import shutil
import os
from PIL import Image
import cv2
import json
import requests
from collections import defaultdict
import pandas as pd
from ultralytics import YOLO


app = Flask(__name__)
app.secret_key = '18dj395'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# List to store the names of resized images
resized_images = []

@app.route('/')
def home():
    # Read the data from Object_Report.csv
    # output = capture_print_output()
    report_path = 'Object_Report_Overvw.csv'
    if os.path.exists(report_path):
        df = pd.read_csv(report_path)
        # Convert the dataframe to HTML table
        table = df.to_html(index=False)
    else:
        table = None

    return render_template('home.html', resized_images=resized_images, table=table)


@app.route('/upload', methods=['POST'])
def upload_files():
    proj_addr = request.form.get('address')
    selected_room = request.form.get('room')
    uploaded_files = request.files.getlist("file[]")
    iffiles = request.files['file[]'].filename

    if proj_addr and selected_room and iffiles != '':
        session['sel_room'] = selected_room
        session['proj_addr'] = proj_addr
        flash(f'Your files uploaded successfully.', 'upload')
        uploaded_filenames = []
        session['delete_files'] = ''
        for file in uploaded_files:
            if file:
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                uploaded_filenames.append(file.filename)
                # Resize the image
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                image = Image.open(image_path)
                new_width = 640
                new_height = 320
                resized_image = image.resize((new_width, new_height))
                # Define the resized image path
                resized_folder = 'uploads/resized'
                resized_image_path = os.path.join(resized_folder, file.filename)
                # Save the resized image
                resized_image.save(resized_image_path)
                # Add the name of the resized image to the list
                resized_images.append(file.filename)
    else:
        if iffiles == '':
            flash(f'Be sure to select one or more image files', 'upload')
        if not proj_addr or not selected_room:
              flash(f'A project Address & Room selection are required!', 'upload')
        return redirect(url_for('dashboard'))

    session['uploaded_files'] = uploaded_filenames
    return redirect(url_for('dashboard'))


# @app.route('/move-images')
def move_images():
    source_dir = 'runs/detect/predict'
    destination_dir = 'static/saved_images'

    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # List all files in the source directory
    files = os.listdir(source_dir)

    # Filter for image files (optional, based on your requirements)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Move each file to the destination directory
    for file in image_files:
        source_path = os.path.join(source_dir, file)
        destination_path = os.path.join(destination_dir, file)
        shutil.move(source_path, destination_path)

    # After moving the files, delete the 'predict' folder
    shutil.rmtree(source_dir)

    return jsonify({"message": f"Moved {len(image_files)} images and deleted the 'predict' folder."})


# Function to delete image files
def delete_image_files():
    # Delete files in the uploads folder
    uploads_folder = 'uploads'
    for filename in os.listdir(uploads_folder):
        file_path = os.path.join(uploads_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Delete files in the uploads/resized folder
    resized_folder = 'uploads/resized'
    for filename in os.listdir(resized_folder):
        file_path = os.path.join(resized_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Delete files in the static/saved_images folder
    resized_folder = 'static/saved_images'
    for filename in os.listdir(resized_folder):
        file_path = os.path.join(resized_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Delete File *******************************************
    file_path = "Object_Report.csv"

    try:
        os.remove(file_path)
        print("File deleted successfully.")
    except FileNotFoundError:
        print("File not found.")
    except PermissionError:
        print("Permission denied. Make sure you have the necessary permissions to delete the Object_Report.csv file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Delete File *******************************************
    file_path = "Object_Report_Overvw.csv"

    try:
        os.remove(file_path)
        print("Object_Report_Overvw deleted successfully.")
    except FileNotFoundError:
        print("File not found.")
    except PermissionError:
        print("Permission denied. Make sure you have the necessary permissions to delete the Object_Report_Overvw.csv file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Delete File *******************************************
    file_path = "Object_Report_Totals.csv"

    try:
        os.remove(file_path)
        print("Object_Report_Totals deleted successfully.")
    except FileNotFoundError:
        print("File not found.")
    except PermissionError:
        print(
            "Permission denied. Make sure you have the necessary permissions to delete the Object_Report_Totals.csv file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Delete File *******************************************
    file_path = "reports/HRG_SCOPE_REPORT.csv"

    try:
        os.remove(file_path)
        print("reports/HRG_SCOPE_REPORT deleted successfully.")
    except FileNotFoundError:
        print("File not found.")
    except PermissionError:
        print(
            "Permission denied. Make sure you have the necessary permissions to delete the Object_Report_Totals.csv file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Delete File *******************************************
    file_path = "Object_Report_Complete.csv"

    try:
        os.remove(file_path)
        print("Object_Report_Complete deleted successfully.")
    except FileNotFoundError:
        print("File not found.")
    except PermissionError:
        print(
            "Permission denied. Make sure you have the necessary permissions to delete the Object_Report_Totals.csv file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    session.pop('image_urls', None)
    session.pop('sel_room', None)
    session.pop('proj_addr', None)
    session.pop('hrg_template', None)
    session.pop('csv_complete', None)

    # return render_template('dashboard.html', image_urls=image_urls)



# Add the delete_image_files function to the code
@app.route('/delete', methods=['POST'])
def delete_files():
    delete_image_files()
    session['delete_files'] = 'files deleted'

    return redirect(url_for('dashboard'))

@app.route('/clear', methods=['POST'])
def clear_files():
    clear_image_files()
    # session['delete_files'] = 'files deleted'

    return redirect(url_for('dashboard'))

def clear_image_files():
    # Delete files in the static/saved_images folder
    saved_folder = 'static/saved_images'
    for filename in os.listdir(saved_folder):
        file_path = os.path.join(saved_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Delete files in the uploads folder
    uploads_folder = 'uploads'
    for filename in os.listdir(uploads_folder):
        file_path = os.path.join(uploads_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Delete files in the uploads/resized folder
    resized_folder = 'uploads/resized'
    for filename in os.listdir(resized_folder):
        file_path = os.path.join(resized_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Delete File *******************************************
    file_path = "Object_Report_Overvw.csv"

    try:
        os.remove(file_path)
        print("File deleted successfully.")
    except FileNotFoundError:
        print("File not found.")
    except PermissionError:
        print(
            "Permission denied. Make sure you have the necessary permissions to delete the Object_Report_Overvw.csv file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    session.pop('uploaded_files', None)
    session['image_urls'] = None
    # session.pop['image_urls', None]


@app.route('/detect', methods=['POST'])
def detect_objects():
    # Call the image_detection function
    image_detection()
    return redirect(url_for('dashboard'))

@app.route('/detect', methods=['POST'])
def image_detection():
    class_lst = ['door', 'ceilinglight', 'window', 'outlet', 'cabinet', 'lightswitch', 'ceilingfan', 'blinds', 'sink',
                 'tree', 'yard', 'closet', 'vanity', 'mirror', 'toilet', 'fridge', 'garagedoor', 'fence', 'furnace',
                 'kitchenrange', 'shower', 'fireplace', 'dishwasher', 'waterheater', 'deck', 'microwave',
                 'garagedooropener', 'clothesdryer', 'AC', 'clotheswasher', 'shed', 'gate', 'porchlight', 'sumppump']

    folder_path = 'uploads/resized'
    # folder_path = '/home/kwayne/PycharmProjects/Flask_Website/uploads/resized/'
    files = os.listdir(folder_path)
    num_images = len(files)

    # Run inference on each image
    model = YOLO("housing_model-2024-01-16.pt")
    results = model.predict(source=folder_path, save=True)
    # results = model.predict(source=folder_path)
    # Create an empty dataframe
    df = pd.DataFrame(columns=['Image'] + class_lst)
    j = 0
    while j < num_images:
        r = results[j]  # only one result as only one image was inferred
        class_names = r.names
        img_pth = r.path.split('/')

        # Initialize the dictionary with zeros for all object classes
        image_dict_classes = defaultdict(int)
        for obj_class in class_lst:
            image_dict_classes[obj_class] = 0

        for b in r.boxes:
            ten_str = str(b.cls)
            ten_str = int(ten_str[8:-3])
            obj_class = class_names[ten_str]
            # obj_class = obj['name']
            image_dict_classes[obj_class] += 1
            obj_class = ''

        row = {'Image': img_pth[7]}
        row.update(image_dict_classes)
        df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
        j += 1

    # Replace NaN values with zeros
    df.fillna(0, inplace=True)

    # Print the resulting dataframe
    csv_file = 'Object_Report.csv'
    df.to_csv(csv_file, index=False)
    print(df)
    move_images()
    data_report()



def image_detection_old():
    url = "https://api.ultralytics.com/v1/predict/K3yIPEs0S2kYpJ1h2BSh"
    headers = {"x-api-key": "66c354f9eaa5a98b6c9ad54978429cf4c3919c452b"}
    data = {"size": 640, "confidence": 0.25, "iou": 0.45}

    class_lst = ['door', 'ceilinglight', 'window', 'outlet', 'cabinet', 'lightswitch', 'ceilingfan', 'blinds', 'sink',
                 'tree', 'yard', 'closet', 'vanity', 'mirror', 'toilet', 'fridge', 'garagedoor', 'fence', 'furnace',
                 'kitchenrange', 'shower', 'fireplace', 'dishwasher', 'waterheater', 'deck', 'microwave',
                 'garagedooropener', 'clothesdryer', 'AC', 'clotheswasher', 'shed', 'gate', 'porchlight', 'sumppump']

    folder_path = 'uploads/resized'
    output_path = 'static/saved_images'
    files = os.listdir(folder_path)

    # Create an empty dataframe
    df = pd.DataFrame(columns=['Image'] + class_lst)

    # Run inference on each image
    for file in files:
        img_file = f'{folder_path}/{file}'
        image_dict_classes = defaultdict(int)

        # Initialize the dictionary with zeros for all object classes
        for obj_class in class_lst:
            image_dict_classes[obj_class] = 0

        with open(img_file, "rb") as f:
            response = requests.post(url, headers=headers, data=data, files={"image": f})

        # Check for successful response
        response.raise_for_status()

        # Print inference results
        # json_str = json.dumps(response.json(), indent=2)

        # json_obj = json.loads(json_str)
        # Parse the inference results
        json_obj = json.loads(response.content)

        # ==========================================
        bounding_boxes = []
        names = []
        for key, value in json_obj.items():
            if key == 'data':
                for obj in value:
                    labels = obj['name']
                    h = obj['height']
                    w = obj['width']
                    x = obj['xcenter']
                    y = obj['ycenter']
                    bounding_box = (x, y, w, h)
                    names.append(labels)
                    bounding_boxes.append(bounding_box)
        output_img = f'{output_path}/{file}'
        save_image_with_bounding_boxes(img_file, bounding_boxes, names, output_img)
        # print(file, labels, bounding_boxes)

        # ========================================

        # Count the occurrences of each object class
        for key, value in json_obj.items():
            if isinstance(value, list):
                for obj in value:
                    # obj_class = value[0]['name']
                    obj_class = obj['name']
                    image_dict_classes[obj_class] += 1

        row = {'Image': file}
        row.update(image_dict_classes)
        df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)

    # Replace NaN values with zeros
    df.fillna(0, inplace=True)

    # Print the resulting dataframe
    csv_file = 'Object_Report.csv'
    df.to_csv(csv_file, index=False)
    print(df)
    data_report()


def data_report():
    # Specify the path and filename for the CSV file
    csv_file = 'Object_Report.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    df['Totals'] = df.drop('Image', axis=1).sum(axis=1)
    df = df[['Image', 'Totals']]

    df.to_csv('Object_Report_Overvw.csv', index=False)
    man_col_ttls()

def man_col_ttls():
    csv_file = 'Object_Report.csv'
    df = pd.read_csv(csv_file)

    # Sum all but the first column
    columns = ['Totals', ]
    column_sums = df.iloc[:, 1:].sum().tolist()
    for i in column_sums:
        columns.append(i)

    df.loc[len(df)] = columns

    # Calculate the sum for each row, excluding the first column
    row_totals = df.iloc[:, 1:].sum(axis=1)

    # Add the row totals as a new column to the dataframe
    df['Totals'] = row_totals

    # Add the room name row to the DataFrame
    if 'sel_room' in session:
        room_selection = session['sel_room']
    else:
        flash(f'Scan failed! Upload new Images and select room.', 'room_sel')
        return redirect(url_for('dashboard'))


    room_sel = room_selection.split(" ")  # Splitting the string by space
    str_len = len(room_sel)

    rm_row = []
    for r in room_sel:
        rm_row.append(r)

    i = 0
    while i < 36 - str_len:
        rm_row.append('----')
        i += 1

    df.loc[-1] = rm_row

    df.index = df.index + 1
    df = df.sort_index()

    # Add an address row to the DataFrame
    # addr = '302 East 123rd Terr'
    addr = session['proj_addr']
    addr = addr.split(" ")
    str_len = len(addr)

    addr_row = []
    for a in addr:
        addr_row.append(a)

    i = 0
    while i < 36 - str_len:
        addr_row.append('----')
        i += 1

    df.loc[-1] = addr_row

    df.index = df.index + 1
    df = df.sort_index()

    # Print the updated dataframe
    csv_file2 = 'Object_Report_Totals.csv'
    csv_file3 = 'Object_Report_Complete.csv'
    df.to_csv(csv_file2, index=False)
    # xxx
    # sess_csv_comp = session.get('csv_complete')
    if 'csv_complete' in session:
        sess_csv_comp = session.get('csv_complete')
        print(f'Message in session for template:  {sess_csv_comp}')
        # csv_tmp = 'Object_Report_Complete.csv'
        df_comp = pd.read_csv(csv_file3)
        df_comp = df_comp.fillna(0)
        df_comp = pd.concat([df, df_comp])
        df_comp.to_csv(csv_file3, index=False)
    else:
        df.to_csv(csv_file3, index=False)
        session['csv_complete'] = 'yes'
    # xxx
    print(df)
    image_display()
    # return redirect(url_for('dashboard'))

def save_image_with_bounding_boxes(image_path, bounding_boxes, names, output_img):
    # Load the image
    # print(f'save {output_img}')
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    print(image_path, names, bounding_boxes)
    # Draw bounding boxes on the image
    i = 0
    for (x, y, w, h) in bounding_boxes:
        # print(f'COOR-1: {x}, {y}, {w}, {h}')
        name = str(names[i])
        i += 1
        x = int(x * width)
        y = int(y * height)
        w = int(w * width)
        h = int(h * height)
        x, y, w, h = int(x), int(y), int(w), int(h)  # Convert to integers
        x = x - 25
        y = y - 35
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the image with bounding boxes
    cv2.imwrite(output_img, image)

def scope_report():
    # Load object report
    csv_file = 'Object_Report_Totals.csv'
    obj_report = pd.read_csv(csv_file)
    obj_report = obj_report.iloc[[0, 1, -1]].reset_index(drop=True)
    # Load class list
    c_list = ['door', 'ceilinglight', 'window', 'outlet', 'cabinet', 'lightswitch', 'ceilingfan', 'blinds', 'sink',
                 'tree', 'yard', 'closet', 'vanity', 'mirror', 'toilet', 'fridge', 'garagedoor', 'fence', 'furnace',
                 'kitchenrange', 'shower', 'fireplace', 'dishwasher', 'waterheater', 'deck', 'microwave',
                 'garagedooropener', 'clothesdryer', 'AC', 'clotheswasher', 'shed', 'gate', 'porchlight', 'sumppump']

    # Read the CSV file and store it in a DataFrame
    # sess_tmp = session.get('hrg_template')
    if 'hrg_template' in session:
        sess_tmp = session.get('hrg_template')
        print(f'Message in session for template:  {sess_tmp}')
        csv_temp = 'reports/HRG_SCOPE_REPORT.csv'
        df = pd.read_csv(csv_temp)
    else:
        csv_tmp = 'resources/FUll_HRG_Template_Zero_Quanity-Beta_master.csv'
        df = pd.read_csv(csv_tmp)
        df = df.fillna('----')
        csv_temp = 'reports/HRG_SCOPE_REPORT.csv'
        df.to_csv(csv_temp, index=False)
        session['hrg_template'] = 'yes'

    prj_addr = str(obj_report.loc[0, 'Image']) + ' ' + str(obj_report.loc[0, 'door']) + ' ' + str(obj_report.loc[0, 'ceilinglight']) + ' ' + str(obj_report.loc[0, 'window']) + ' ' + str(obj_report.loc[0, 'outlet'])
    room_sel = str(obj_report.loc[1, 'Image']) + ' ' + str(obj_report.loc[1, 'door']) + ' ' + str(obj_report.loc[1, 'ceilinglight'])
    prj_addr = prj_addr.replace('nan', '')
    room_sel = room_sel.replace('nan', '')
    room_sel = room_sel.replace('----', '').strip()
    print(f'================= Room Selection: {room_sel} ======================')

    if room_sel == 'Master Bedroom':
        master_bedroom(obj_report, c_list, df)
    if room_sel == 'Main Floor Bath':
        main_floor_bath(obj_report, c_list, df)
    if room_sel == 'Kitchen':
        kitchen(obj_report, c_list, df)
    if room_sel == 'Multi-Room':
        multi_room(obj_report, c_list, df)

def master_bedroom(obj_report, c_list, df):
    blinds = int(obj_report.loc[2, 'blinds'])
    window = int(obj_report.loc[2, 'window'])
    door = int(obj_report.loc[2, 'door'])
    ceilingfan = int(obj_report.loc[2, 'ceilingfan'])
    ceilinglight = int(obj_report.loc[2, 'ceilinglight'])
    outlet = int(obj_report.loc[2, 'outlet'])
    lightswitch = int(obj_report.loc[2, 'lightswitch'])
    closet = int(obj_report.loc[2, 'closet'])
    # print(f'blinds: {blinds} | window: {window} | door: {door} | ceilingfan: {ceilingfan} | ceilinglight: {ceilinglight} | outlet: {outlet} | lightswitch: {lightswitch} | closet: {closet}')

    if blinds > 0:
        df.loc[1322, 'Quantity'] = blinds
    if window > 0:
        df.loc[1326, 'Quantity'] = window
    # if door > 0:
    #   df.loc[1335, 'Quantity'] = door
    if ceilingfan > 0:
        df.loc[1378, 'Quantity'] = ceilingfan
    if ceilinglight > 0:
        df.loc[1381, 'Quantity'] = ceilinglight
    if outlet > 0:
        df.loc[1395, 'Quantity'] = outlet
    if lightswitch > 0:
        df.loc[1394, 'Quantity'] = lightswitch
    if closet > 0:
        df.loc[1358, 'Quantity'] = closet

    # Save the dataframe to a CSV file
    out_path = 'reports/HRG_SCOPE_REPORT.csv'
    df.to_csv(out_path, index=True)

def main_floor_bath(obj_report, c_list, df):
    door = int(obj_report.loc[2, 'door'])
    closet = int(obj_report.loc[2, 'closet'])
    outlet = int(obj_report.loc[2, 'outlet'])
    lightswitch = int(obj_report.loc[2, 'lightswitch'])
    vanity = int(obj_report.loc[2, 'vanity'])
    sink = int(obj_report.loc[2, 'sink'])
    shower = int(obj_report.loc[2, 'shower'])
    toilet = int(obj_report.loc[2, 'toilet'])
    # bathtub = obj_report.loc[1, 'bathtub']
    ceilinglight = int(obj_report.loc[2, 'ceilinglight'])

    if door > 0:
        df.loc[953, 'Quantity'] = door
    if ceilinglight > 0:
        df.loc[988, 'Quantity'] = ceilinglight
    if outlet > 0:
        df.loc[1000, 'Quantity'] = outlet
    if lightswitch > 0:
        df.loc[997, 'Quantity'] = lightswitch
    if closet > 0:
        df.loc[975, 'Quantity'] = closet
    if vanity > 0:
        df.loc[1028, 'Quantity'] = vanity
    if sink > 0:
        df.loc[1040, 'Quantity'] = sink
    if shower > 0:
        df.loc[1030, 'Quantity'] = shower
    if toilet > 0:
        df.loc[1036, 'Quantity'] = toilet
    # Save the dataframe to a CSV file
    out_path = 'reports/HRG_SCOPE_REPORT.csv'
    df.to_csv(out_path, index=False)

def kitchen(obj_report, c_list, df):
    door = int(obj_report.loc[2, 'door'])
    closet = int(obj_report.loc[2, 'closet'])
    sink = int(obj_report.loc[2, 'sink'])
    fridge = int(obj_report.loc[2, 'fridge'])
    dishwasher = int(obj_report.loc[2, 'dishwasher'])
    ceilinglight = int(obj_report.loc[2, 'ceilinglight'])
    outlet = int(obj_report.loc[2, 'outlet'])
    lightswitch = int(obj_report.loc[2, 'lightswitch'])
    blinds = int(obj_report.loc[2, 'blinds'])
    window = int(obj_report.loc[2, 'window'])
    cabinet = int(obj_report.loc[2, 'cabinet'])
    kitchenrange = int(obj_report.loc[2, 'kitchenrange'])

    if door > 0:
        df.loc[567, 'Quantity'] = door
    if closet > 0:
        df.loc[586, 'Quantity'] = closet
    if sink > 0:
        df.loc[592, 'Quantity'] = sink
    if fridge > 0:
        df.loc[618, 'Quantity'] = fridge
    if dishwasher > 0:
        df.loc[621, 'Quantity'] = dishwasher
    if ceilinglight > 0:
        df.loc[639, 'Quantity'] = ceilinglight
    if outlet > 0:
        df.loc[655, 'Quantity'] = outlet
    if lightswitch > 0:
        df.loc[653, 'Quantity'] = lightswitch
    if blinds > 0:
        df.loc[659, 'Quantity'] = blinds
    if window > 0:
        df.loc[665, 'Quantity'] = window
    if cabinet > 0:
        df.loc[677, 'Quantity'] = cabinet
    if kitchenrange > 0:
        df.loc[620, 'Quantity'] = kitchenrange

    # Save the dataframe to a CSV file
    out_path = 'reports/HRG_SCOPE_REPORT.csv'
    df.to_csv(out_path, index=False)


def multi_room(obj_report, c_list, df):
    door = int(obj_report.loc[2, 'door'])
    df.loc[264, 'Quantity'] = int(door)
    closet = int(obj_report.loc[2, 'closet'])
    df.loc[285, 'Quantity'] = int(closet)
    blinds = int(obj_report.loc[2, 'blinds'])
    df.loc[371, 'Quantity'] = int(blinds)
    outlet = int(obj_report.loc[2, 'outlet'])
    df.loc[354, 'Quantity'] = int(outlet)
    lightswitch = int(obj_report.loc[2, 'lightswitch'])
    df.loc[351, 'Quantity'] = int(lightswitch)
    vanity = int(obj_report.loc[2, 'vanity'])
    df.loc[257, 'Quantity'] = int(vanity)
    sink = int(obj_report.loc[2, 'sink'])
    df.loc[255, 'Quantity'] = int(sink)
    fridge = int(obj_report.loc[2, 'fridge'])
    df.loc[315, 'Quantity'] = int(fridge)
    shower = int(obj_report.loc[2, 'shower'])
    df.loc[241, 'Quantity'] = int(shower)
    toilet = int(obj_report.loc[2, 'toilet'])
    df.loc[243, 'Quantity'] = int(toilet)
    ceilinglight = int(obj_report.loc[2, 'ceilinglight'])
    df.loc[336, 'Quantity'] = int(ceilinglight)
    ceilingfan = int(obj_report.loc[2, 'ceilingfan'])
    df.loc[335, 'Quantity'] = int(ceilingfan)
    window = int(obj_report.loc[2, 'window'])
    df.loc[362, 'Quantity'] = int(window)
    cabinet = int(obj_report.loc[2, 'cabinet'])
    df.loc[389, 'Quantity'] = int(cabinet)
    kitchenrange = int(obj_report.loc[2, 'kitchenrange'])
    df.loc[317, 'Quantity'] = int(kitchenrange)
    """
    if door > 0:
        df.loc[264, 'Quantity'] = int(door)
    if closet > 0:
        df.loc[285, 'Quantity'] = int(closet)
    if blinds > 0:
        df.loc[371, 'Quantity'] = int(blinds)
    if outlet > 0:
        df.loc[354, 'Quantity'] = int(outlet)
    if lightswitch > 0:
        df.loc[351, 'Quantity'] = int(lightswitch)
    if vanity > 0:
        df.loc[257, 'Quantity'] = int(vanity)
    if sink > 0:
        df.loc[255, 'Quantity'] = int(sink)
    if fridge > 0:
        df.loc[315, 'Quantity'] = int(fridge)
    if shower > 0:
        df.loc[241, 'Quantity'] = int(shower)
    if toilet > 0:
        df.loc[243, 'Quantity'] = int(toilet)
    if ceilinglight > 0:
        df.loc[336, 'Quantity'] = int(ceilinglight)
    if ceilingfan > 0:
        df.loc[335, 'Quantity'] = int(ceilingfan)
    if window > 0:
        df.loc[362, 'Quantity'] = int(window)
    if cabinet > 0:
        df.loc[389, 'Quantity'] = int(cabinet)
    if kitchenrange > 0:
        df.loc[317, 'Quantity'] = int(kitchenrange)
    """
    # Save the dataframe to a CSV file
    out_path = 'reports/HRG_SCOPE_REPORT.csv'
    df.to_csv(out_path, index=False)


# @app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['POST'])
def login():
     if request.method == 'POST':
         username = request.form['username']
         password = request.form['password']
         # Perform authentication logic here (e.g., check username and password against a database)
         if username == 'admin' and password == 'scope&scan':
             session['logged_in'] = True  # Set the session variable
             return redirect(url_for('dashboard'))
         else:
             return render_template('home.html', error='Invalid credentials')
     return render_template('home.html')



@app.route('/dashboard')
def dashboard():
    if session.get('logged_in'):
        # Read the contents of the Object_Report.csv file
        report_path = 'Object_Report_Overvw.csv'
        if os.path.isfile(report_path):
            try:
                df = pd.read_csv(report_path)
                # Convert the dataframe to HTML table
                table = df.to_html(index=False)
                print("File read successfully.")
            except PermissionError:
                print("Permission denied. Make sure you have the necessary permissions to read the file.")
            except Exception as e:
                print(f"An error occurred while reading the file: {str(e)}")
        else:
            print("File not found.")
            table = []

        # Render the dashboard template and pass the table to it

        delete_files = session.get('delete_files', [])
        if delete_files == 'files deleted':
            uploaded_files = []
        else:
            uploaded_files = session.get('uploaded_files', [])
        return render_template('dashboard.html', uploaded_files=uploaded_files, table=table)
    else:
        # User is not logged in, redirect to the login page
        return redirect(url_for('home'))


@app.route('/download_csv', methods=['POST'])
def download_csv():
    filename = 'Object_Report_Complete.csv'  # Specify the name of the CSV file
    return send_file(filename, as_attachment=True)

@app.route('/download_csv2', methods=['POST'])
def download_csv2():
    filename = 'reports/HRG_SCOPE_REPORT.csv'  # Specify the name of the CSV file
    return send_file(filename, as_attachment=True)


@app.route('/image_display', methods=['GET'])
def image_display():
    image_folder = 'static/saved_images'
    # image_folder = 'runs/detect/predict'
    image_urls = []

    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            image_path = image_path.replace('\\', '/')
            image_path = image_path.replace('static/', '')
            print(image_path)
            image_urls.append(image_path)

    # Store the variable in the session
    # print(image_urls)
    session['image_urls'] = image_urls

    # Flash a success message
    flash('Scan process Completed!', 'scan')
    scope_report()
    return redirect('/dashboard')




if __name__ == '__main__':
    app.secret_key = '18dj395'
    app.run(debug=True)