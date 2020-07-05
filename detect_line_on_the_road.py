import cv2
import matplotlib.pylab as plt
import numpy as np














def roi(img,vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,vertices,255)
    masked_img = cv2.bitwise_and(img,mask)
    return masked_img



def Draw_the_line(img,lines):
    img = np.copy(img)
    blank_image =np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)

    Lcord1 = ()
    Lcord2 = ()
    Rcord1 = ()
    Rcord2 = ()

    try:
        right = 0 #counter
        left = 0 #counter
        Rline = []
        Lline = []
        Horline = []
        for line in lines:

            for x1,y1,x2,y2 in line: # {x1 ve y1 this is one line's DOWN coordinate} {x2 and y2 one line's UP coordinate  x2 ve y2 üst nokta


                if abs(y2 - y1) > 15 and abs(x2-x1)>7  : ##  side line

                    if x2 - 700 <0 and x1-700 < 0: # left lines
                        left += 1
                        cordLBas = [x1,y1] # up possible coordinates
                        cordLBit = [x2,y2] # down possible  coordinates
                        Lline.append([cordLBas,cordLBit])#the possible coordinates append the Left Line list


                    if x2-700 >0 and x1-700>0: #right line
                        right += 1
                        cordRBas = [x1,y1] # up coordinates
                        cordRBit = [x2,y2] # down coordinates
                        Rline.append([cordRBas,cordRBit])#the possible coordinates append the right Line list

                elif abs(y2-y1) < 0 and 150 <abs(x2-x1) < 400: ### horizontal lines
                    cordHBas = [x1, y1]
                    cordHBit = [x2, y2]
                    Horline.append([cordHBas, cordHBit])







        try:

            #######     this process detecting possible  max and min coordinates for RİGHT line #######

            x1 = [] # X axis up coordinates list
            y1 = [] # Y axis up coordinates list
            x2 = [] # X axis down coordinates list
            y2 = [] # y axis down coordinates list
            for line1 in Rline:   # for exp  RLine = [[x1,y1,x2,y2],[x1,y1,x2,y2],[x1,y1,x2,y2]........how much line]
                x1.append(line1[0][0]) # x1 list to  append  x axis up coordinate for all lines
                y1.append(line1[0][1]) # y1 list to  append  y axis up coordinate for all lines
                x2.append(line1[1][0]) # x2 list to  append  x axis down coordinate for all lines
                y2.append(line1[1][1]) # y2 list to  append  x axis down coordinate for all lines

            Rcord1 = (min(x1),min(y1)) # as you know as x1 and y1 is one line's up coordinate
            #the right line should be right side on the image so should be  x's coordinates more than the middle point
            Rcord2 = (max(x2),max(y2))

            print('Rcord1:',Rcord1,'other:',Rcord2 )

            cv2.line(blank_image,(Rcord1),(Rcord2),[0,0,  255],10) # add right line  to blank image
            cv2.putText(blank_image,"Right Line",(Rcord2),cv2.FONT_HERSHEY_SIMPLEX,1.5,[0,0,255],3)

                #######     this process detecting possible  max and min coordinates for LEFT line #######

            x10 = []
            y10 = []
            x20 = []
            y20 = []
            try:
                for line1 in Lline:
                    x10.append(line1[0][0])
                    y10.append(line1[0][1])
                    x20.append(line1[1][0])
                    y20.append(line1[1][1])
                Lcord1 = (min(x10),max(y10))
                Lcord2 = (max(x20),min(y20))
            except:
                pass

            cv2.line(blank_image,(Lcord1),(Lcord2),[0,255,  255],10) # add left line  to blank image
            cv2.putText(blank_image,"Left Line",(Lcord1),cv2.FONT_HERSHEY_SIMPLEX,1.5,[0,255,255],3) # add text  to blank image




            ###### this process detect to horizontal stop lines####
            #### I am actually I don't think necassary this process####
            ## so I did disable ####

            x11 = []
            y11 = []
            x21 = []
            y21 = []
            for line1 in Horline:
                x11.append(line1[0][0])
                y11.append(line1[0][1])
                x21.append(line1[1][0])
                y21.append(line1[1][1])

            #cord11 = (min(x11), max(y11))
            #cord21 = (max(x21), min(y21)) #this is included process (optionally)

            #cv2.line(blank_image, (cord11), (cord21), [0, 0, 255], 5) # (optionally)



            ##### CLEAR THE LİST ######
            'The purpose of this process is to clear all coordinate information in each frame.' \
            ' otherwise, the cooridates of the previous frame affect the coordinates of the current frame'
            clear_the_list = [x1, y1, x2, y2, x11, y11, x21, y21, x10, y10, x20, y20,Rline,Lline]
            for i in clear_the_list:
                i.clear()

        except Exception as e:
            print("[ERROR MESAGE 1] :",str(e)) # this process bad affect to your system performance
                                            # you may use pass method here  (optionally)










        print(f"detected possible right lines :{right}\ndetected possible left lines :{left} ") #count

        img =cv2.addWeighted(img,0.8,blank_image,1,0.0)
        # this process adds the over processed blank image to the original image

        return img,Lcord1, Rcord2
    except Exception as e:
        print("[ERROR MESAGE 2] :",str(e)) # this process bad affect to your system performance
                                        # you may use pass method here  (optionally)



def prossece_img(image):


    up_threshold = 435 #this method specifies the up threshold on the frame

    down_threshold = 640  #this method specifies the down threshold on the frame

    img_vertices = [(0,down_threshold), (200, down_threshold), (500, up_threshold), (830, up_threshold), (1150, down_threshold), (200, down_threshold)]
    # upper method include image region on the original image
    gray_img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    ker = np.ones((20, 10), dtype=np.float32) / 1  # necessary for smooth method

    smooth = cv2.filter2D(gray_img, -5, ker)  # for line noise reduce
    canny_img = cv2.Canny(gray_img,150,250) # detect to all edge on the image and convert to black and white frame with only edge



    gaus_img = cv2.GaussianBlur(canny_img,(5,5),900000000000000000000)

    cutting_img = roi(gaus_img,np.array([img_vertices],np.int32))

    lines = cv2.HoughLinesP(cutting_img,rho=1,theta=np.pi/30,threshold=200,lines=np.array([]),
                                minLineLength=60,maxLineGap=5)
    # this upper  method prediction  possible line on the frame


    with_lines_img,Left_cord,Right_cord= Draw_the_line(image,lines) #this method sends possible lines to draw_the_line method
    # and returns image with line and coordinates


    print("left down coordinate :",Left_cord,"\nright down coordinate :",Right_cord)

    cv2.line(with_lines_img,(683,600),(683,760),[0,255,255],10) #drawing line on the image with returned coordinates

    rgb_plt_img = cv2.cvtColor(with_lines_img,cv2.COLOR_BGR2RGB) #convert gray image to rgb image

#    return rgb_plt_img # for the video frame. Disable now

    plt.imshow(rgb_plt_img)

    plt.show()
def Run_with_image(path=r"C:\Users\pc2\Desktop\python\py\ets2_218.png"):
    image = cv2.imread(f"{path}")
    prossece_img(image)
def Run_with_video_frame(path='etss.mp4'):

    video = cv2.VideoCapture(f"{path}")

    while True:
        ret,frame = video.read()

        prosseced_img = prossece_img(frame)

        cv2.imshow("window", prosseced_img)
        if cv2.waitKey(15) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break



if __name__ == "__main__":
    Run_with_image()
    cv2.waitKey(0)
    cv2.destroyAllWindows()