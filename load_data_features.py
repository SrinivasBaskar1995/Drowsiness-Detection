#function to load the dataset from txt file
def load(is_train=True,use_three=True):

    seq_len = 5 #seq len for LSTM training
    flag=True
    #if use_first:
    if is_train:
        path = '../data_txt/data-train.txt'
    else:
        path = '../data_txt/data-test.txt'
    if use_three:
        flag=False
    else:
        flag=True
    
    data = []
    labels = []
    
    curr = None
    data_point = []
    data_name = []
    
    f = open(path, "r")
    contents = f.read().split("\n")
    f.close()
    count=0
    
    #reading the data points from txt file and loading it into a array in the form of sequences of 5
    for content in contents:
        points = content.split(",")
        if len(points)!=10:
            continue    
        if int(points[len(points)-1])==5 and flag:
            continue
        if curr==None:
            curr = points[0].split("_")[0]
            count=0
        elif curr!=points[0].split("_")[0]:
            curr = points[0].split("_")[0]
            data_point=[]
            data_name=[]
            count=0
        
        if len(data_point)<seq_len:
            temp = []
            for i in range(1,len(points)-1):
                temp.append(float(points[i]))
            data_point.append(temp)
            data_name.append(points[0])
        else:
            data_point.pop(0)
            temp = []
            for i in range(1,len(points)-1):
                temp.append(float(points[i]))
            data_point.append(temp)
            data_name.pop(0)
            data_name.append(points[0])
            
        if len(data_point)==seq_len:
            data.append(data_point)
            count+=1
            temp1 = [0,0]
            temp1[int(int(points[0].split("_")[1].split("-")[0])/10)]=1
            labels.append(int(points[len(points)-1])/10)
                
    return data,labels