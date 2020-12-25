prediction<-function(x){
  inp_mat <- matrix(x,byrow = TRUE,nrow = 28,ncol=28)
  inp_mat_array <- array_reshape(inp_mat,c(1, img_rows, img_cols, 1))
  inp_array <- array_reshape(inp,c(1,28,28,1))
  predict(new_model, inp_array)
  ans =  which.max(predict(new_model, inp_array))
  return(ans)
}

