from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
import cv2
import tensorflow as tf
from layers import L1Dist  # Make sure 'layers' module is available
import os
import numpy as np
import pandas as pd

class CamApp(App):
    
    def build(self):
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(text='Verify', on_press=self.verify, size_hint=(1, .1), background_color=(0, 0, 1, 1))
        self.verification_label = Label(text='Verification uninitiated', size_hint=(1, .1))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist': L1Dist})

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout
    
    def update(self, *args):
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture
    
    def preprocess(self, file_path):  # Added 'self' as the first parameter
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img
    def on_button_hover(self, instance, pos):
        if instance.collide_point(*pos):
            instance.background_color = (0.7, 0.7, 1, 1)  # Warna saat hover
        else:
            instance.background_color = (1, 1, 1, 1)  # Warna asli
    
    def verify(self, *args):
        detection_threshold = 0.5
        verification_threshold = 0.5
        
        save_path = os.path.join('application_data', 'input_image', 'input_image.jpg')  # Use lowercase for variable names
        
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(save_path, frame)
        
        # Build results array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            
            # Make Predictions 
            result = self.model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
            results.append(result)
        
        # Detection Threshold: Metric above which a prediction is considered positive 
        detection = np.sum(np.array(results) > detection_threshold)
        
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold
        
        self.verification_label.text = 'verified' if verified else 'unverified'
        
        
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)
        
        if verified:
            # Reshape the results array to 2D
            results_2d = np.squeeze(results, axis=(2,))
            
            # Create a DataFrame with the reshaped results and additional columns
            results_df = pd.DataFrame(results_2d, columns=['Verification Score'])
            results_df['Nama'] = 'haldies'
            results_df['Absen'] = 'hadir'
            
            # Save the DataFrame to an Excel file
            excel_save_path = os.path.abspath(os.path.join('application_data', 'verification_results.xlsx'))
            results_df.to_excel(excel_save_path, index=False)
            
            # Informasi keberhasilan
            Logger.info(f"Verification results saved to {excel_save_path}")
            
            # Tampilkan label atau umpan balik visual
            self.verification_label.text = f"Verified and results saved to {excel_save_path}"
            self.button.background_color = (1, 1, 1, 1)


            
if __name__ == '__main__':
    CamApp().run()
