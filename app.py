from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
import os
from zipfile import ZipFile
import io
import shutil
import os
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename
from collections import defaultdict
import pandas as pd
from ultralytics import YOLO


app = Flask(__name__)
app.secret_key = '18dj395'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESIZED_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'resized')

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
    if 'image' not in request.files:
        flash(f'Only image files accepted!', 'image_accept')
        return jsonify({'redirect': '/dashboard'})
    else:
        proj_addr = request.form.get('address')
        selected_room = request.form.get('room')
        room_nm = request.form.get('rm_name')
        uploaded_files = request.files.getlist("file[]")
        iffiles = request.files['file[]'].filename

        if proj_addr and selected_room and iffiles != '':
            session['sel_room'] = selected_room
            session['proj_addr'] = proj_addr
            session['room_nm'] = room_nm
            flash(f'Your files uploaded successfully.', 'upload')
            uploaded_filenames = []
            session['delete_files'] = ''
            for file in uploaded_files:
                if file.filename == '':
                    continue  # Skip empty files

                filename = secure_filename(file.filename)
                filename = filename.replace(" ", "_")  # Replace spaces with underscores
                uploaded_filenames.append(filename)  # Add the processed filename to the list

                # Save the original file (optional)
                original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(original_path)

                # Resize and save the image
                image = Image.open(original_path)
                image = image.resize((640, 320), Image.Resampling.LANCZOS)  # Use Image.Resampling.LANCZOS
                resized_path = os.path.join(app.config['RESIZED_FOLDER'], filename)
                image.save(resized_path)
        else:
            if iffiles == '':
                flash(f'Be sure to select one or more image files', 'upload')
            if not proj_addr or not selected_room:
                  flash(f'A project Address & Room selection are required!', 'upload')
            return redirect(url_for('dashboard'))

    session['uploaded_files'] = uploaded_filenames
    return jsonify({'redirect': '/dashboard'})
    return redirect(url_for('dashboard'))


@app.route('/zip-images')
def zip_images():
    images_folder = 'static/saved_images/'
    output_folder = 'static/zip_file/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    zip_filename = 'scanned_images.zip'
    zip_filepath = os.path.join(output_folder, zip_filename)

    with ZipFile(zip_filepath, 'w') as zip_file:
        for root, dirs, files in os.walk(images_folder):
            if files:
                for file in files:
                    file_path = os.path.join(root, file)
                    print(f"Adding {file_path} to zip")  # Debugging line
                    zip_file.write(file_path, arcname=file)
            else:
                print("No files found in the directory.")  # Debugging line

    return f'Images have been zipped and saved to {zip_filepath}'


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

    # Delete Zip File *******************************************
    file_path = "static/zip_file/scanned_images.zip"

    try:
        os.remove(file_path)
        print("scanned_images.zip File deleted successfully.")
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
    session.pop('room_nm', None)
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
    room_name = session['room_nm'].split(" ")
    str_len2 = len(room_name)
    str_len = len(room_sel)
    if str_len2 == 0:
        room_rw = room_sel
    else:
        room_rw = room_name
        str_len = str_len2

    rm_row = []
    for r in room_rw:
        rm_row.append(r)

    i = 0
    while i < 36 - str_len:
        rm_row.append('____')
        i += 1

    df.loc[-1] = rm_row

    df.index = df.index + 1
    df = df.sort_index()

    # Add an address row to the DataFrame
    addr = session['proj_addr']
    addr = addr.split(" ")
    str_len = len(addr)

    addr_row = []
    for a in addr:
        addr_row.append(a)

    i = 0
    while i < 36 - str_len:
        addr_row.append('____')
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
    zip_images()
    image_display()
    # return redirect(url_for('dashboard'))

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
        df = df.fillna('0')
        csv_temp = 'reports/HRG_SCOPE_REPORT.csv'
        df.to_csv(csv_temp, index=False)
        session['hrg_template'] = 'yes'

    prj_addr = str(obj_report.loc[0, 'Image']) + ' ' + str(obj_report.loc[0, 'door']) + ' ' + str(obj_report.loc[0, 'ceilinglight']) + ' ' + str(obj_report.loc[0, 'window']) + ' ' + str(obj_report.loc[0, 'outlet'])
    room_sel = str(obj_report.loc[1, 'Image']) + ' ' + str(obj_report.loc[1, 'door']) + ' ' + str(obj_report.loc[1, 'ceilinglight'])
    prj_addr = prj_addr.replace('nan', '')
    room_sel = room_sel.replace('nan', '')
    room_sel = room_sel.replace('____', '').strip()
    print(f'================= Room Selection: {room_sel} ======================')

    # if room_sel == 'Master Bedroom':
    #     master_bedroom(obj_report, c_list, df)
    # if room_sel == 'Main Floor Bath':
    #     main_floor_bath(obj_report, c_list, df)
    # if room_sel == 'Kitchen':
    #     kitchen(obj_report, c_list, df)
    if room_sel == 'Multi-Room':
        multi_room(obj_report, c_list, df)


# INTERIOR ===========================
    blinds = int(obj_report.loc[2, 'blinds'])
    df.loc[379, 'Quantity'] = int(blinds)  # xx

    cabinet = int(obj_report.loc[2, 'cabinet'])
    df.loc[387, 'Quantity'] = int(cabinet)  # xx

    ceilingfan = int(obj_report.loc[2, 'ceilingfan'])
    df.loc[345, 'Quantity'] = int(ceilingfan)  # xx

    ceilinglight = int(obj_report.loc[2, 'ceilinglight'])
    df.loc[349, 'Quantity'] = int(ceilinglight)  # xx

    closet = int(obj_report.loc[2, 'closet'])
    df.loc[293, 'Quantity'] = int(closet)  # xx

    dishwasher = int(obj_report.loc[2, 'dishwasher'])
    df.loc[328, 'Quantity'] = int(dishwasher)  # xx

    door = int(obj_report.loc[2, 'door'])
    df.loc[264, 'Quantity'] = int(door)

    fridge = int(obj_report.loc[2, 'fridge'])
    df.loc[325, 'Quantity'] = int(fridge)  # xx

    kitchenrange = int(obj_report.loc[2, 'kitchenrange'])
    df.loc[327, 'Quantity'] = int(kitchenrange)  # xx

    lightswitch = int(obj_report.loc[2, 'lightswitch'])
    df.loc[361, 'Quantity'] = int(lightswitch)  # xx

    mirror = int(obj_report.loc[2, 'mirror'])
    df.loc[246, 'Quantity'] = int(mirror)  # xx

    outlet = int(obj_report.loc[2, 'outlet'])
    df.loc[362, 'Quantity'] = int(outlet)  # xx

    shower = int(obj_report.loc[2, 'shower'])
    df.loc[251, 'Quantity'] = int(shower)  # xx

    sink = int(obj_report.loc[2, 'sink'])
    df.loc[262, 'Quantity'] = int(sink)  # xx

    toilet = int(obj_report.loc[2, 'toilet'])
    df.loc[252, 'Quantity'] = int(toilet)  # xx

    vanity = int(obj_report.loc[2, 'vanity'])
    df.loc[262, 'Quantity'] = int(vanity)  # xx

    window = int(obj_report.loc[2, 'window'])
    df.loc[362, 'Quantity'] = int(window)  # ??

    # UTILITY: ================================
    clothesdryer = int(obj_report.loc[2, 'clothesdryer'])
    df.loc[362, 'Quantity'] = int(clothesdryer)

    clotheswasher = int(obj_report.loc[2, 'clotheswasher'])
    df.loc[362, 'Quantity'] = int(clotheswasher)

    furnace = int(obj_report.loc[2, 'furnace'])
    df.loc[239, 'Quantity'] = int(furnace)  # xx

    garagedooropener = int(obj_report.loc[2, 'garagedooropener'])
    df.loc[204, 'Quantity'] = int(garagedooropener)  # xx

    sumppump = int(obj_report.loc[2, 'sumppump'])
    df.loc[362, 'Quantity'] = int(sumppump)

    waterheater = int(obj_report.loc[2, 'waterheater'])
    df.loc[362, 'Quantity'] = int(waterheater)  # xx

    # EXTERIOR: ================================
    AC = int(obj_report.loc[2, 'AC'])
    df.loc[362, 'Quantity'] = int(AC)

    deck = int(obj_report.loc[2, 'deck'])
    df.loc[362, 'Quantity'] = int(deck)

    fence = int(obj_report.loc[2, 'fence'])
    df.loc[174, 'Quantity'] = int(fence)  # xx

    garagedoor = int(obj_report.loc[2, 'garagedoor'])
    df.loc[204, 'Quantity'] = int(garagedoor)  # xx

    gate = int(obj_report.loc[2, 'gate'])
    df.loc[362, 'Quantity'] = int(gate)

    porchlight = int(obj_report.loc[2, 'porchlight'])
    df.loc[76, 'Quantity'] = int(porchlight)  # xx

    shed = int(obj_report.loc[2, 'shed'])
    df.loc[362, 'Quantity'] = int(shed)

    tree = int(obj_report.loc[2, 'tree'])
    df.loc[67, 'Quantity'] = int(tree)  # xx

    yard = int(obj_report.loc[2, 'yard'])
    df.loc[6, 'Quantity'] = int(yard)  # xx

    # Save the dataframe to a CSV file
    out_path = 'reports/HRG_SCOPE_REPORT.csv'
    df.to_csv(out_path, index=False)


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


# @app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['POST'])
def login():
     if request.method == 'POST':
         username = request.form['username']
         password = request.form['password']
         # Perform authentication logic here (e.g., check username and password against a database)
         if username == 'admin' and password == 'scope&scan':
             session['logged_in'] = True  # Set the session variable
             delete_image_files()
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
    try:
        return send_file(filename, as_attachment=True)
        print("File downloaded successfully.")
    except FileNotFoundError:
        print("File not found.")
        flash('File Not Found!', 'nofile_image')
        return redirect('/dashboard')


@app.route('/download_csv2', methods=['POST'])
def download_csv2():
    filename = 'reports/HRG_SCOPE_REPORT.csv'  # Specify the name of the CSV file

    try:
        return send_file(filename, as_attachment=True)
        print("File downloaded successfully.")
    except FileNotFoundError:
        print("File not found.")
        flash('File Not Found!', 'nofile_scope')
        return redirect('/dashboard')

@app.route('/download_zip', methods=['GET'])
def download_zip():
    path_to_zip = "static/zip_file/scanned_images.zip"
    if os.path.exists(path_to_zip):
        return send_file(path_to_zip, as_attachment=True)
    else:
        print("File not found.")
        flash('File Not Found!', 'zip_download')
        return redirect('/dashboard')


@app.route('/image_display', methods=['GET'])
def image_display():
    image_folder = 'static/saved_images'
    # image_folder = 'runs/detect/predict'
    image_urls = []

    try:
        for filename in os.listdir(image_folder):
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
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
    except FileNotFoundError as e:
        print(f"An error occurred: {e}")
        flash('No Image Files found!', 'display')



if __name__ == '__main__':
    app.secret_key = '18dj395'
    app.run(debug=True)