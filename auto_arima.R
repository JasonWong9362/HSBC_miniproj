# import data
folder_path<-"C:\Users\jeson\project\MiniProject\master" 
file_name<-"/integrated_candlesticks3600.csv" 
file_path<-paste(folder_path,file_name,sep="") # create the file_path

data <- read.csv(file = 'integrated_candlesticks3600.csv')
head(data)


