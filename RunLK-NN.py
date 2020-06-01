import numpy as np
from keras.models import model_from_yaml
import cv2

# import matplotlib.pyplot as plt


###############################################################

''' Method to load and initialize our model and weights.'''


def loadNN(model_file, weights_file):
    # load YAML and create model
    yaml_file = open(model_file, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights(weights_file)
    print("Loaded model from disk")
    return loaded_model


###############################################################

''' Method to process images and then predict the LK vector using the
model.'''


def LK(frame1, frame2, model):
    # process input images
    # read in both input images
    img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # =cv2.IMREAD_GRAYSCALE(image1_path)
    img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)  # =cv2.IMREAD_GRAYSCALE(image2_path)

    # stack them together vertically
    nn_input = np.concatenate((img1, img2), axis=0)  # vertical stacking with axis=0, horiz is axis=1

    # resize the new stacked image
    nn_input.resize((28, 28))
    nn_input = np.reshape(nn_input, (1, 28, 28, -1))

    # predict the LK vector for this input
    nn_output = model.predict(nn_input)
    print(nn_output)
    LKvector = [nn_output[0][1] - nn_output[0][0], nn_output[0][3] -
                nn_output[0][2], nn_output[0][5] - nn_output[0][4]]
    print(LKvector)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot([0, LKvector[0]], [0, LKvector[1]], [0, LKvector[2]])
    return LKvector


###############################################################

recording = []
mod = loadNN('model.yaml', 'model.h5')

cap = cv2.VideoCapture(1)
ret1, frame1 = cap.read()
ret2, frame2 = cap.read()
while (ret1):
    # swap frames to be ready for next frame input
    frame1, frame2 = frame2, frame1

    # Capture frame-by-frame
    ret, frame2 = cap.read()

    recording.append(LK(frame1, frame2, mod))
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    # cv2.imshow(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()