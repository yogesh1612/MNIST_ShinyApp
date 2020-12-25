show_pixels(
  round(as.vector(t(data[1,,,]))),
  grid = c(28, 28),
  size = c(200, 200)
 # params = list(fill = list(color = "#FF3388"))
)


temp <- get_pixels()
show_pixels(temp,
            grid = c(28,28),
            size = c(200,200))

temp_mat <- matrix(temp,byrow = TRUE,nrow = 28,ncol=28)

show_pixels(
    round(as.vector(t(temp_mat))),
    grid = c(28, 28),
    size = c(200, 200)
    # params = list(fill = list(color = "#FF3388"))
  )
