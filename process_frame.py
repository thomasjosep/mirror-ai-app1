import os
import time
import cv2
import numpy as np
from utils import find_angle, get_landmark_features, draw_text, draw_dotted_line
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.exceptions import GoogleCloudError
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# Check if Firebase app is already initialized
try:
    db = firestore.client()
except Exception:
    cred_json = os.getenv("FIREBASE_CREDENTIAL_JSON")
    if not cred_json:
        raise Exception("FIREBASE_CREDENTIAL_JSON environment variable not set")
    cred_dict = json.loads(cred_json)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    db = firestore.client()


class ProcessFrame:
    def __init__(self, thresholds, flip_frame = False):
        
        # Set if frame should be flipped or not.
        self.flip_frame = flip_frame

        # self.thresholds
        self.thresholds = thresholds

        # Font type.
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # line type
        self.linetype = cv2.LINE_AA

        # set radius to draw arc
        self.radius = 20

        # Colors in BGR format.
        self.COLORS = {
                        'blue'       : (0, 127, 255),
                        'red'        : (255, 50, 50),
                        'green'      : (0, 255, 127),
                        'light_green': (100, 233, 127),
                        'yellow'     : (255, 255, 0),
                        'magenta'    : (255, 0, 255),
                        'white'      : (255,255,255),
                        'cyan'       : (0, 255, 255),
                        'light_blue' : (102, 204, 255)
                      }



        # Dictionary to maintain the various landmark features.
        self.dict_features = {}
        self.left_features = {
                                'shoulder': 11,
                                'elbow'   : 13,
                                'wrist'   : 15,                    
                                'hip'     : 23,
                                'knee'    : 25,
                                'ankle'   : 27,
                                'foot'    : 31
                             }

        self.right_features = {
                                'shoulder': 12,
                                'elbow'   : 14,
                                'wrist'   : 16,
                                'hip'     : 24,
                                'knee'    : 26,
                                'ankle'   : 28,
                                'foot'    : 32
                              }

        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0

        
        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
            'state_seq': [],

            'start_inactive_time': time.perf_counter(),
            'start_inactive_time_front': time.perf_counter(),
            'INACTIVE_TIME': 0.0,
            'INACTIVE_TIME_FRONT': 0.0,

            # 0 --> Bend Backwards, 1 --> Bend Forward, 2 --> Keep shin straight, 3 --> Deep squat
            'DISPLAY_TEXT' : np.full((4,), False),
            'COUNT_FRAMES' : np.zeros((4,), dtype=np.int64),

            'LOWER_HIPS': False,

            'INCORRECT_POSTURE': False,

            'prev_state': None,
            'curr_state':None,

            'SQUAT_COUNT': 0,
            'IMPROPER_SQUAT':0,
            'TOTAL_SQUATS': 0 
            
        }
        
        self.FEEDBACK_ID_MAP = {
                                0: ('BEND BACKWARDS', 215, (0, 153, 255)),
                                1: ('BEND FORWARD', 215, (0, 153, 255)),
                                2: ('KNEE FALLING OVER TOE', 170, (255, 80, 80)),
                                3: ('SQUAT TOO DEEP', 125, (255, 80, 80))
                               }
        
        self.squats_ref = db.collection('squats')

        self.last_count_time = time.time()
        self.COUNT_COOLDOWN = 1.0 
    
    def check_db_connection(self):
        try:
            # Set a shorter timeout for the connection check
            timeout = 10  # 10 seconds timeout
            self.squats_ref.limit(1).get(timeout=timeout)
            print("✅ Successfully connected to Firestore database")
            return True
        except Exception as e:
            print(f"❌ Firebase connection error: {str(e)}")
            # Check common connection issues
            if "failed to connect to all addresses" in str(e):
                print("📡 Network connectivity issue - Please check your internet connection")
            elif "Permission denied" in str(e):
                print("🔑 Authentication error - Please verify your credentials file")
            elif "DEADLINE_EXCEEDED" in str(e):
                print("⌛ Connection timed out - Server might be unreachable")
            return False
        
    # def store_squat_counts(self):
    #     """Stores current squat counts to Firestore"""
    #     try:
    #         # Get current timestamp
    #         timestamp = firestore.SERVER_TIMESTAMP
            
    #         # Create data dictionary
    #         squat_data = {
    #             'correct_count': self.state_tracker['SQUAT_COUNT'],
    #             'incorrect_count': self.state_tracker['IMPROPER_SQUAT'],
    #             'timestamp': timestamp,
    #             'session_id': str(int(time.time()))  # Unique session identifier
    #         }
            
    #         # Update document with latest counts
    #         self.squats_ref.document('latest_counts').set(squat_data)
    #         print("✅ Successfully updated squat counts in database")
            
    #     except Exception as e:
    #         print(f"❌ Error storing counts: {str(e)}")
    def store_squat_counts(self):
        """Stores current squat counts and suggestions to Firestore"""
        try:
            # Get current timestamp
            timestamp = firestore.SERVER_TIMESTAMP
            
            # Get active suggestions
            suggestions = []
            # Add form check suggestions
            for idx in np.where(self.state_tracker['DISPLAY_TEXT'])[0]:
                suggestions.append(self.FEEDBACK_ID_MAP[idx][0])
            
            # Add hip position suggestion if needed
            if self.state_tracker['LOWER_HIPS']:
                suggestions.append('LOWER YOUR HIPS')
            
            # Create data dictionary
            squat_data = {
                'correct_count': self.state_tracker['SQUAT_COUNT'],
                'incorrect_count': self.state_tracker['IMPROPER_SQUAT'], 
                'timestamp': timestamp,
                'session_id': str(int(time.time())),  # Unique session identifier
                'suggestions': suggestions  # Add collected suggestions
            }
            
            # Update document with latest counts and suggestions
            self.squats_ref.document('latest_counts').set(squat_data)
            print("✅ Successfully updated squat counts and suggestions in database")
            
        except Exception as e:
            print(f"❌ Error storing data: {str(e)}")

        


    def _get_state(self, knee_angle):
        """Determine the state based on knee angle"""
        if knee_angle is None:
            return None
        # Adjust thresholds for better state detection    
        if 0 <= knee_angle <= 20:  # Standing
            return 's1'
        elif 20 < knee_angle <= 70:  # Going down
            return 's2'
        elif knee_angle > 70:  # Deep squat
            return 's3'
        return None

    def _update_state_sequence(self, state):
        """Updates the state sequence for squat tracking"""
        print(f"Current State: {state}")  # Debug print
        print(f"Current Sequence: {self.state_tracker['state_seq']}")  # Debug print
        
        if state is None:
            return
            
        # State machine logic
        if len(self.state_tracker['state_seq']) == 0:
            if state == 's2':  # Start with going down
                self.state_tracker['state_seq'].append(state)
                print("Starting new squat - Adding s2")
        
        elif len(self.state_tracker['state_seq']) == 1:
            if state == 's3' and self.state_tracker['state_seq'][-1] == 's2':
                self.state_tracker['state_seq'].append(state)
                print("Adding s3 - Squat position reached")
        
        elif len(self.state_tracker['state_seq']) == 2:
            if state == 's1':  # Standing position after complete sequence
                # Ready to count the squat
                print("Complete squat detected - Ready to count")

            


    def _show_feedback(self, frame, c_frame, dict_maps, lower_hips_disp):


        if lower_hips_disp:
            draw_text(
                    frame, 
                    'LOWER YOUR HIPS', 
                    pos=(30, 80),
                    text_color=(0, 0, 0),
                    font_scale=0.6,
                    text_color_bg=(255, 255, 0)
                )  

        for idx in np.where(c_frame)[0]:
            draw_text(
                    frame, 
                    dict_maps[idx][0], 
                    pos=(30, dict_maps[idx][1]),
                    text_color=(255, 255, 230),
                    font_scale=0.6,
                    text_color_bg=dict_maps[idx][2]
                )

        return frame



    def process(self, frame: np.array, pose):
        play_sound = None
       

        frame_height, frame_width, _ = frame.shape

        # Process the image.
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            ps_lm = keypoints.pose_landmarks

            nose_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
            left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)
            right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)

            offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord)

            if offset_angle > self.thresholds['OFFSET_THRESH']:
                
                display_inactivity = False

                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time

                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.thresholds['INACTIVE_THRESH']:
                    self.state_tracker['SQUAT_COUNT'] = 0
                    self.state_tracker['IMPROPER_SQUAT'] = 0
                    display_inactivity = True

                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7, self.COLORS['magenta'], -1)

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                if display_inactivity:
                    # cv2.putText(frame, 'Resetting SQUAT_COUNT due to inactivity!!!', (10, frame_height - 90), 
                    #             self.font, 0.5, self.COLORS['blue'], 2, lineType=self.linetype)
                    play_sound = 'reset_counters'
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                draw_text(
                    frame, 
                    "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), 
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )  
                

                draw_text(
                    frame, 
                    "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), 
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),
                    
                )  
                
                
                draw_text(
                    frame, 
                    'CAMERA NOT ALIGNED PROPERLY! STAY SIDE FACED', 
                    pos=(30, frame_height-60),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                ) 
                
                
                draw_text(
                    frame, 
                    'OFFSET ANGLE: '+str(offset_angle), 
                    pos=(30, frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                ) 

                # Reset inactive times for side view.
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0
                self.state_tracker['prev_state'] =  None
                self.state_tracker['curr_state'] = None
            
            # Camera is aligned properly.
            else:

                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter()


                dist_l_sh_hip = abs(left_foot_coord[1]- left_shldr_coord[1])
                dist_r_sh_hip = abs(right_foot_coord[1] - right_shldr_coord)[1]

                shldr_coord = None
                elbow_coord = None
                wrist_coord = None
                hip_coord = None
                knee_coord = None
                ankle_coord = None
                foot_coord = None

                if dist_l_sh_hip > dist_r_sh_hip:
                    shldr_coord = left_shldr_coord
                    elbow_coord = left_elbow_coord
                    wrist_coord = left_wrist_coord
                    hip_coord = left_hip_coord
                    knee_coord = left_knee_coord
                    ankle_coord = left_ankle_coord
                    foot_coord = left_foot_coord

                    multiplier = -1
                                     
                
                else:
                    shldr_coord = right_shldr_coord
                    elbow_coord = right_elbow_coord
                    wrist_coord = right_wrist_coord
                    hip_coord = right_hip_coord
                    knee_coord = right_knee_coord
                    ankle_coord = right_ankle_coord
                    foot_coord = right_foot_coord

                    multiplier = 1
                    

                # ------------------- Verical Angle calculation --------------
                
                hip_vertical_angle = find_angle(shldr_coord, np.array([hip_coord[0], 0]), hip_coord)
                cv2.ellipse(frame, hip_coord, (30, 30), 
                            angle = 0, startAngle = -90, endAngle = -90+multiplier*hip_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

                draw_dotted_line(frame, hip_coord, start=hip_coord[1]-80, end=hip_coord[1]+20, line_color=self.COLORS['blue'])




                knee_vertical_angle = find_angle(hip_coord, np.array([knee_coord[0], 0]), knee_coord)
                cv2.ellipse(frame, knee_coord, (20, 20), 
                            angle = 0, startAngle = -90, endAngle = -90-multiplier*knee_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3,  lineType = self.linetype)

                draw_dotted_line(frame, knee_coord, start=knee_coord[1]-50, end=knee_coord[1]+20, line_color=self.COLORS['blue'])



                ankle_vertical_angle = find_angle(knee_coord, np.array([ankle_coord[0], 0]), ankle_coord)
                cv2.ellipse(frame, ankle_coord, (30, 30),
                            angle = 0, startAngle = -90, endAngle = -90 + multiplier*ankle_vertical_angle,
                            color = self.COLORS['white'], thickness = 3,  lineType=self.linetype)

                draw_dotted_line(frame, ankle_coord, start=ankle_coord[1]-50, end=ankle_coord[1]+20, line_color=self.COLORS['blue'])

                # ------------------------------------------------------------
        
                
                # Join landmarks.
                cv2.line(frame, shldr_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, wrist_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, shldr_coord, hip_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, knee_coord, hip_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, knee_coord,self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, foot_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                
                # Plot landmark points
                cv2.circle(frame, shldr_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, elbow_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, wrist_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, hip_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, knee_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, ankle_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, foot_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)

                

                current_state = self._get_state(int(knee_vertical_angle))
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)



                # -------------------------------------- COMPUTE COUNTERS --------------------------------------

                # Replace the counter logic section with:
                if current_state == 's1' and len(self.state_tracker['state_seq']) == 2:
                    current_time = time.time()
                    if current_time - self.last_count_time >= self.COUNT_COOLDOWN:
                        self.state_tracker['TOTAL_SQUATS'] += 1
                        
                        if not self.state_tracker['INCORRECT_POSTURE']:
                            self.state_tracker['SQUAT_COUNT'] += 1
                            print(f"\n✅ Correct Squat! Count: {self.state_tracker['SQUAT_COUNT']}")
                        else:
                            self.state_tracker['IMPROPER_SQUAT'] += 1
                            print(f"\n❌ Incorrect Squat! Count: {self.state_tracker['IMPROPER_SQUAT']}")
                        
                        print(f"📊 Total Squats: {self.state_tracker['TOTAL_SQUATS']}")
                        
                        # Update Streamlit session state
                        if 'squat_count' not in st.session_state:
                            st.session_state['squat_count'] = 0
                        if 'improper_count' not in st.session_state:
                            st.session_state['improper_count'] = 0
                        if 'total_count' not in st.session_state:
                            st.session_state['total_count'] = 0
                            
                        st.session_state['squat_count'] = self.state_tracker['SQUAT_COUNT']
                        st.session_state['improper_count'] = self.state_tracker['IMPROPER_SQUAT']
                        st.session_state['total_count'] = self.state_tracker['TOTAL_SQUATS']
                        
                        # Reset for next squat
                        self.last_count_time = current_time
                        self.state_tracker['state_seq'] = []
                        self.state_tracker['INCORRECT_POSTURE'] = False
                        
                        # Store counts in database
                        self.store_squat_counts()


                # ----------------------------------------------------------------------------------------------------




                # -------------------------------------- PERFORM FEEDBACK ACTIONS --------------------------------------

                else:
                    if hip_vertical_angle > self.thresholds['HIP_THRESH'][1]:
                        self.state_tracker['DISPLAY_TEXT'][0] = True
                        

                    elif hip_vertical_angle < self.thresholds['HIP_THRESH'][0] and \
                         self.state_tracker['state_seq'].count('s2')==1:
                            self.state_tracker['DISPLAY_TEXT'][1] = True
                        
                                        
                    
                    if self.thresholds['KNEE_THRESH'][0] < knee_vertical_angle < self.thresholds['KNEE_THRESH'][1] and \
                       self.state_tracker['state_seq'].count('s2')==1:
                        self.state_tracker['LOWER_HIPS'] = True


                    elif knee_vertical_angle > self.thresholds['KNEE_THRESH'][2]:
                        self.state_tracker['DISPLAY_TEXT'][3] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True

                    
                    if (ankle_vertical_angle > self.thresholds['ANKLE_THRESH']):
                        self.state_tracker['DISPLAY_TEXT'][2] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True


                # ----------------------------------------------------------------------------------------------------


                
                
                # ----------------------------------- COMPUTE INACTIVITY ---------------------------------------------

                display_inactivity = False
                
                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:

                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']
                    self.state_tracker['start_inactive_time'] = end_time

                    # When counters reset due to inactivity
                    if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                        self.state_tracker['SQUAT_COUNT'] = 0
                        self.state_tracker['IMPROPER_SQUAT'] = 0
                        display_inactivity = True
                        self.store_squat_counts()  # Add this line

                
                else:
                    
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0

                # -------------------------------------------------------------------------------------------------------
              


                hip_text_coord_x = hip_coord[0] + 10
                knee_text_coord_x = knee_coord[0] + 15
                ankle_text_coord_x = ankle_coord[0] + 10

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    hip_text_coord_x = frame_width - hip_coord[0] + 10
                    knee_text_coord_x = frame_width - knee_coord[0] + 15
                    ankle_text_coord_x = frame_width - ankle_coord[0] + 10

                
                
                if 's3' in self.state_tracker['state_seq'] or current_state == 's1':
                    self.state_tracker['LOWER_HIPS'] = False

                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']]+=1

                frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP, self.state_tracker['LOWER_HIPS'])



                if display_inactivity:
                    # cv2.putText(frame, 'Resetting COUNTERS due to inactivity!!!', (10, frame_height - 20), self.font, 0.5, self.COLORS['blue'], 2, lineType=self.linetype)
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0

                
                cv2.putText(frame, str(int(hip_vertical_angle)), (hip_text_coord_x, hip_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(knee_vertical_angle)), (knee_text_coord_x, knee_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(ankle_vertical_angle)), (ankle_text_coord_x, ankle_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)

                 
                # In the process method, update the display section:
                draw_text(
                    frame, 
                    "TOTAL: " + str(st.session_state.get('total_count', self.state_tracker['TOTAL_SQUATS'])), 
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(0, 128, 255)
                )  

                draw_text(
                    frame, 
                    "CORRECT: " + str(st.session_state.get('squat_count', self.state_tracker['SQUAT_COUNT'])), 
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )  

                draw_text(
                    frame, 
                    "INCORRECT: " + str(st.session_state.get('improper_count', self.state_tracker['IMPROPER_SQUAT'])), 
                    pos=(int(frame_width*0.68), 130),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0)
                )
                
                
                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0    
                self.state_tracker['prev_state'] = current_state
                                  

       
        
        else:

            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']

            display_inactivity = False

            # When counters reset due to inactivity
            if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                self.state_tracker['SQUAT_COUNT'] = 0
                self.state_tracker['IMPROPER_SQUAT'] = 0
                display_inactivity = True
                self.store_squat_counts()  # Add this line

            self.state_tracker['start_inactive_time'] = end_time

            draw_text(
                    frame, 
                    "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), 
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )  
                

            draw_text(
                    frame, 
                    "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), 
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),
                    
                )  

            if display_inactivity:
                play_sound = 'reset_counters'
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0
            
            
            # Reset all other state variables
            
            self.state_tracker['prev_state'] =  None
            self.state_tracker['curr_state'] = None
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full((5,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros((5,), dtype=np.int64)
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()
            
            
            
        return frame, play_sound


