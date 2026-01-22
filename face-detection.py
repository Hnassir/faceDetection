import streamlit as st
import cv2 as cv
import time
#from datetime import datetime



def detection(colors,minneighbors,scaleFactor,save,i):

    #model of face detection
    cascade=cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    #videocapture class open the camera
    cap=cv.VideoCapture(0) # cap is our webcam object

    image_placeholder=st.empty()


    while True:
            
        ret,frame=cap.read()

        if not ret:
            st.warning('failed to capture frame')
            break
            
        #changing frame color to gray
        gray_frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

        #after detection it returns coordonnees x,y height and width
        face_res=cascade.detectMultiScale(gray_frame,scaleFactor,minneighbors)

        for (x,y,w,h) in face_res:
            cv.rectangle(frame,(x,y),(x+w,y+h),colors,2)

        #2- timestamp !!
            
        if save :
            save=False
            cv.imwrite(f'saved_image_{i}.jpg',frame)
            text=st.success(f"Image saved successfully!")
            time.sleep(3)
            text.empty()

        #we switch image color
        rgb_frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)

        #using a placeholder to show the webcam's window
        image_placeholder.image(rgb_frame)
        
        if cv.waitKey(1) & 0xFF == ord('q') :
            image_placeholder.empty()
            break

    cap.release()
    cv.destroyAllWindows()



def main():

    if 'count' not in st.session_state:
        st.session_state.count=0

    st.title('Face detection tool ')
    st.write("Press the button below to start detecting faces from your webcam")

    color_val=st.color_picker('choose color',value='#00FF00')

    R=color_val[1:3]
    G=color_val[3:5]
    B=color_val[5:]

    R=int(R,base=16)
    G=int(G,base=16)
    B=int(B,base=16)

    min_n=st.slider('choose min Neighbors value',min_value=1,max_value=5,value=2,step=1)
    scaleF=st.slider('choose scale factor value',min_value=0.5,max_value=2.,value=1.1,step=0.1)

    with st.container(border=True,horizontal=True):

        cam=st.button('start the experience !')
        save=st.button('press to save the frame image')

        if cam :
            detection((B,G,R),min_n,scaleF,save,0)

        if save :
            if 'count' in st.session_state:
                st.session_state.count+=1
            detection((B,G,R),min_n,scaleF,save,st.session_state.count)


if __name__=='__main__' :
    main()