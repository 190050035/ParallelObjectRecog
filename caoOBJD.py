import cv2
import ray
import time

ray.init()

@ray.remote
def detect_objects(classNames, img):

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 225, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 225, 0), 2)
    return img

def process_image_files(file_paths):
    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    start_time_serial = time.time()


    processed_images_serial = []
    for file in file_paths:
        img = cv2.imread(file)
        img_processed = detect_objects.remote(classNames, img)
        img_processed = ray.get(img_processed)
        processed_images_serial.append(img_processed)

    end_time_serial = time.time()
    print(f"Serial Time: {end_time_serial - start_time_serial} seconds")

    start_time_parallel = time.time()


    results = [detect_objects.remote(classNames, cv2.imread(file)) for file in file_paths]
    processed_images_parallel = ray.get(results)

    end_time_parallel = time.time()
    print(f"Parallel Time: {end_time_parallel - start_time_parallel} seconds")


    for img_processed in processed_images_parallel:
        cv2.imshow("Output", img_processed)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

image_files = [
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\12283150_12d37e6389_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\25691390_f9944f61b5_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\262985539_1709e54576_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\1045023827_4ec3e8ba5c_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\2383514521_1fc8d7b0de_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\2502287818_41e4b0c4fb_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\2516944023_d00345997d_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\3132016470_c27baa00e8_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\3627527276_6fe8cd9bfe_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\3651581213_f81963d1dd_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\3800883468_12af3c0b50_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\3862500489_6fd195d183_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\3878153025_8fde829928_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\4410436637_7b0ca36ee7_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\4782628554_668bc31826_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\5951960966_d4e1cda5d0_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\6584515005_fce9cec486_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\6821351586_59aa0dc110_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\7581246086_cf7bbb7255_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\7933423348_c30bd9bd4e_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\8053677163_d4c8f416be_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\8239308689_efa6c11b08_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\8433365521_9252889f9a_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\8512296263_5fc5458e20_z.jpg',
    'C:\\Users\\Md.Khalid\\PycharmProjects\\objectdetection\\images\\8699757338_c3941051b6_z.jpg',

]
process_image_files(image_files)
