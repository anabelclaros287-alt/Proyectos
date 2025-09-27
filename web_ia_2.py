import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import tensorflow as tf

def main():
    print('-web-')
    st.title('clasificador de imagenes de plantas')
    st.write('cargar una imagen y predice ')

    file = st.file_uploader('carga una imagen')
    if file:
        image = Image.open(file)
        # opsoleto -> use_column_width=True
        st.image(image, use_container_width=True) 
        resized_image = image.resize((300,300))
        # modo rgb
        #img_array = np.array(resized_image) / 255
        #convierto a bgr
        img_array = np.array(resized_image)
        img_array = img_array[:, :, ::-1]
        img_array = img_array.reshape((1,300,300,3))


        model = tf.keras.models.load_model('modelo_plantas_limon_h5.h5')

        predictions = model.predict(img_array)
        print(model.input_shape)
        print(predictions)
        print(max(predictions))
        print(predictions.argmax())

        cifar10_class = ['saludable', 'enferma']

        fig,ax = plt.subplots()
        y_pos = np.arange(len(cifar10_class))
        ax.barh(y_pos, predictions[0], align = 'center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cifar10_class)

        ax.set_xlabel('probabilidad: ')
        ax.set_title('predicion de imagen')
        st.pyplot(fig)
    
    else:
        st.text('no has cargado una imagen')

main()