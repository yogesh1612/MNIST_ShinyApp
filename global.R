predictionss<-function(loaded_model,x){
  #new_model <- load_model_hdf5("mnist_model.h5")
  #print(sum(x))
  inp_mat <- matrix(x,byrow = TRUE,nrow = 28,ncol=28)
  #print(sum(inp_mat))
  inp_mat_array <- array_reshape(inp_mat,c(1, 28, 28, 1))
  #inp_array <- array_reshape(inp,c(1,28,28,1))
 # predict(new_model, inp_array)
  ans<-loaded_model%>%predict_classes(inp_mat_array)
  #print(ans)
  return(ans)
}

