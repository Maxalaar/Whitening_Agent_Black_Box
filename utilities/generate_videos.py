import cv2


def post_rendering_processing(images):
    new_width = 246
    ratio = images.shape[1] / images.shape[0]
    new_height = int(new_width / ratio)
    resize_images = cv2.resize(images, (new_width, new_height))
    gray_images = cv2.cvtColor(resize_images, cv2.COLOR_RGB2GRAY)

    return gray_images


def generate_video(images, output_video_path, fps=30):
    if len(images) == 0:
        print("The list of images is empty.")
        return

    height, width = images[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path+'.mp4', fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print("Error: Could not open the video writer.")
        return

    for image in images:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_image)

    video_writer.release()