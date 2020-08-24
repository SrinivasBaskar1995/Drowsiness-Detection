import numpy as np
import math

#function to calculate euclidean distance between two points
def dist(a,b):
    return math.sqrt(math.pow(a[0]-b[0],2)+math.pow(a[1]-b[1],2))

#function to calculate the features of left eye
def features_left(landmarks):
    EAR = (dist(landmarks[38],landmarks[42])+dist(landmarks[39],landmarks[41]))/(2*dist(landmarks[37],landmarks[40]))
    MAR = dist(landmarks[63],landmarks[67])/dist(landmarks[61],landmarks[65])
    area = float(math.pow(dist(landmarks[38],landmarks[41])/2,2))*np.pi
    perimeter = dist(landmarks[37],landmarks[38])+dist(landmarks[38],landmarks[39])+dist(landmarks[39],landmarks[40])+dist(landmarks[40],landmarks[41])+dist(landmarks[41],landmarks[42])+dist(landmarks[42],landmarks[37])
    PUC = (4*np.pi*area)/(math.pow(perimeter,2))
    MOE = MAR/EAR
    FA = dist(landmarks[34],landmarks[9])
    return EAR,MAR,PUC,MOE,FA

#function to calculate the features of right eye
def features_right(landmarks):
    EAR = (dist(landmarks[44],landmarks[48])+dist(landmarks[45],landmarks[47]))/(2*dist(landmarks[43],landmarks[46]))
    MAR = dist(landmarks[63],landmarks[67])/dist(landmarks[61],landmarks[65])
    area = float(math.pow(dist(landmarks[44],landmarks[47])/2,2))*np.pi
    perimeter = dist(landmarks[43],landmarks[44])+dist(landmarks[44],landmarks[45])+dist(landmarks[45],landmarks[46])+dist(landmarks[46],landmarks[47])+dist(landmarks[47],landmarks[48])+dist(landmarks[48],landmarks[43])
    PUC = (4*np.pi*area)/(math.pow(perimeter,2))
    MOE = MAR/EAR
    FA = dist(landmarks[34],landmarks[9])
    return EAR,MAR,PUC,MOE,FA

path_train = '../data_txt/data_landmarks-train.txt'
path_test = '../data_txt/data_landmarks-test.txt'

paths = [path_train,path_test]

for path in paths:
    f = open(path, "r")
    contents = f.read().split("\n")
    f.close()
    
    data = []
    labels = []
    
    prev_file=None
    normalizers = {} #to store the mean and average of each feature for each video file
    count=0
    initial_EAR_left = []
    initial_PUC_left = []
    initial_MOE_left = []
    initial_EAR_right = []
    initial_PUC_right = []
    initial_MOE_right = []
    initial_MAR = []
    initial_FA = []
    
    #calculating the mean and average for each video file
    for content in contents:
        points = content.split(",")
        filename = points[0].split("_")[0]
        if "0"==points[len(points)-1]:
            if prev_file != filename:
                
                if prev_file!=None:
                    #calculate mean and std for each feature
                    mean_EAR_left = np.mean(initial_EAR_left)
                    std_EAR_left = np.std(initial_EAR_left)
                    mean_PUC_left = np.mean(initial_PUC_left)
                    std_PUC_left = np.std(initial_PUC_left)
                    mean_MOE_left = np.mean(initial_MOE_left)
                    std_MOE_left = np.std(initial_MOE_left)
                    
                    mean_EAR_right = np.mean(initial_EAR_right)
                    std_EAR_right = np.std(initial_EAR_right)
                    mean_PUC_right = np.mean(initial_PUC_right)
                    std_PUC_right = np.std(initial_PUC_right)
                    mean_MOE_right = np.mean(initial_MOE_right)
                    std_MOE_right = np.std(initial_MOE_right)
                    
                    mean_MAR = np.mean(initial_MAR)
                    std_MAR = np.std(initial_MAR)
                    
                    mean_FA = np.mean(initial_FA)
                    std_FA = np.std(initial_FA)
                    
                    normalizers[prev_file] = [(mean_EAR_left,std_EAR_left),(mean_PUC_left,std_PUC_left),(mean_MOE_left,std_MOE_left),(mean_EAR_right,std_EAR_right),(mean_PUC_right,std_PUC_right),(mean_MOE_right,std_MOE_right),(mean_MAR,std_MAR),(mean_FA,std_FA)]
                    
                count=0
                initial_EAR_left = []
                initial_PUC_left = []
                initial_MOE_left = []
                initial_EAR_right = []
                initial_PUC_right = []
                initial_MOE_right = []
                initial_MAR = []
                initial_FA = []
                
                prev_file = filename
                
            if count<3:
                landmarks = []
                for i in range(1,len(points)-1,2):
                    landmarks.append((int(points[i]),int(points[i+1])))
                    
                EAR_left,MAR,PUC_left,MOE_left,FA = features_left(landmarks)
                
                EAR_right,MAR,PUC_right,MOE_right,FA = features_right(landmarks)
                
                initial_EAR_left.append(EAR_left)
                initial_PUC_left.append(PUC_left)
                initial_MOE_left.append(MOE_left)
                initial_EAR_right.append(EAR_right)
                initial_PUC_right.append(PUC_right)
                initial_MOE_right.append(MOE_right)
                initial_MAR.append(MAR)
                initial_FA.append(FA)
                count+=1
                
        elif len(initial_EAR_left)>0:
            #calculate mean and std for each feature
            mean_EAR_left = np.mean(initial_EAR_left)
            std_EAR_left = np.std(initial_EAR_left)
            mean_PUC_left = np.mean(initial_PUC_left)
            std_PUC_left = np.std(initial_PUC_left)
            mean_MOE_left = np.mean(initial_MOE_left)
            std_MOE_left = np.std(initial_MOE_left)
            
            mean_EAR_right = np.mean(initial_EAR_right)
            std_EAR_right = np.std(initial_EAR_right)
            mean_PUC_right = np.mean(initial_PUC_right)
            std_PUC_right = np.std(initial_PUC_right)
            mean_MOE_right = np.mean(initial_MOE_right)
            std_MOE_right = np.std(initial_MOE_right)
            
            mean_MAR = np.mean(initial_MAR)
            std_MAR = np.std(initial_MAR)
            
            mean_FA = np.mean(initial_FA)
            std_FA = np.std(initial_FA)
            
            normalizers[prev_file] = [(mean_EAR_left,std_EAR_left),(mean_PUC_left,std_PUC_left),(mean_MOE_left,std_MOE_left),(mean_EAR_right,std_EAR_right),(mean_PUC_right,std_PUC_right),(mean_MOE_right,std_MOE_right),(mean_MAR,std_MAR),(mean_FA,std_FA)]
            
            count=0
            initial_EAR_left = []
            initial_PUC_left = []
            initial_MOE_left = []
            initial_EAR_right = []
            initial_PUC_right = []
            initial_MOE_right = []
            initial_MAR = []
            initial_FA = []
                    
    print("number of keys : ",len(normalizers.keys()))
        
    #extracting features and normalising them
    for content in contents:
        points = content.split(",")
            
        filename = points[0].split("_")[0]
        if filename not in normalizers.keys():
            continue
        
        landmarks = []
        for i in range(1,len(points)-1,2):
            landmarks.append((int(points[i]),int(points[i+1])))
            
        EAR_left,MAR,PUC_left,MOE_left,FA = features_left(landmarks)
        EAR_right,MAR,PUC_right,MOE_right,FA = features_right(landmarks)
        
        curr_mean_EAR_left = normalizers[filename][0][0]
        curr_std_EAR_left = normalizers[filename][0][1]
        curr_mean_PUC_left = normalizers[filename][1][0]
        curr_std_PUC_left = normalizers[filename][1][1]
        curr_mean_MOE_left = normalizers[filename][2][0]
        curr_std_MOE_left = normalizers[filename][2][1]
        
        curr_mean_EAR_right = normalizers[filename][3][0]
        curr_std_EAR_right = normalizers[filename][3][1]
        curr_mean_PUC_right = normalizers[filename][4][0]
        curr_std_PUC_right = normalizers[filename][4][1]
        curr_mean_MOE_right = normalizers[filename][5][0]
        curr_std_MOE_right = normalizers[filename][5][1]
        
        curr_mean_MAR = normalizers[filename][6][0]
        curr_std_MAR = normalizers[filename][6][1]
        
        curr_mean_FA = normalizers[filename][7][0]
        curr_std_FA = normalizers[filename][7][1]
        
        #normalising features
        EAR_left_norm = (EAR_left - curr_mean_EAR_left)/curr_std_EAR_left
        PUC_left_norm = (PUC_left - curr_mean_PUC_left)/curr_std_PUC_left
        MOE_left_norm = (MOE_left - curr_mean_MOE_left)/curr_std_MOE_left
        
        EAR_right_norm = (EAR_right - curr_mean_EAR_right)/curr_std_EAR_right
        PUC_right_norm = (PUC_right - curr_mean_PUC_right)/curr_std_PUC_right
        MOE_right_norm = (MOE_right - curr_mean_MOE_right)/curr_std_MOE_right
        
        MAR_norm = (MAR - curr_mean_MAR)/curr_std_MAR
        
        FA_norm = (FA - curr_mean_FA)/curr_std_FA
        
        #adding to the data list and label list
        data.append([points[0],(EAR_left+EAR_right)/2,(PUC_left+PUC_right)/2,(MOE_left+MOE_right)/2,MAR,FA,(EAR_left_norm+EAR_right_norm)/2,(PUC_left_norm+PUC_right_norm)/2,(MOE_left_norm+MOE_right_norm)/2,MAR_norm,FA_norm])
        labels.append(int(points[len(points)-1]))
    
    #saving the extracted features to a txt file
    f = open('../data_txt/data-'+path.split("-")[1].split(".")[0]+'.txt', 'w')
    for i in range(len(data)):
        for points in data[i]:
            f.write(str(points))
            f.write(",")
        f.write(str(labels[i]))
        f.write("\n")
    f.close()