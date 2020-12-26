# server <- function(input, output) {
#   output$pixels <- shiny_render_pixels(
#     show_pixels(grid = c(28, 28), size = c(250, 250),
#                brush = matrix(c(0.5,1,1,0.5), 2, 2))
#   )
#   
#   # observeEvent(input$captureDigit, {
#   #   
#   #   #digit_path <- file.path("digits", digit())
#   #   #if (!dir.exists(digit_path)) dir.create(digit_path, recursive = TRUE)
#   #   #saveRDS(input$pixels, paste0(digit_path, "/", as.numeric(Sys.time()), ".rds"))
#   #   
#   #   #digit(floor(runif(1, 1, 10)))
#   #   #output$predicted_number <- input$pixel
#   #   print(input$pixels)
#   #   #agin refresh paint area
#   #   output$pixels <- shiny_render_pixels(
#   #     show_pixels())
#   #   
#   # })
#   observeEvent(input$captureDigit, {
#     #digit_path <- file.path("digits", digit())
#     #if (!dir.exists(digit_path)) dir.create(digit_path, recursive = TRUE)
#     print(sum(input$pixels))
#     
#     #digit(floor(runif(1, 1, 10)))
#     output$pixels <- shiny_render_pixels(
#       show_pixels()
#     )
#   })
#   
#   
# }
# 


server <- function(input, output) {
  
  
  model <- reactive({
    if (is.null(input$file)) { return(NULL) }
    else{
      print(1)
      model <- load_model_hdf5(input$file$datapath)
    }
    return(model)
    
  })
  
  
  
  
  output$model_summary <- renderPrint(
    if (is.null(input$file)) {return("Upload Model to print summary")}
    else{return(summary(model()))}
  
  )
  
  
  output$pixels <- shiny_render_pixels(
    show_pixels(grid = c(28, 28), size = c(250, 250),
                brush = matrix(c(0.5,1,1,0.5), 2, 2))  )
  
  #digit <- reactiveVal(floor(runif(1, 1, 10)))
  # output$prompt <- renderText(paste0("Please draw number ", digit(), ":"))
  
  output$predicted_number<- renderText(
    if(sum(input$pixels)==0){return(NULL)}
    else{
      return(paste0("Predicted Number is :" , predictionss(model(),input$pixels)))
    }
                              
  )
  
    
    

  observeEvent(input$captureDigit, {
    #digit_path <- file.path("digits", digit())
    #if (!dir.exists(digit_path)) dir.create(digit_path, recursive = TRUE)
    #print(1)
    
    #digit(floor(runif(1, 1, 10)))
    output$pixels <- shiny_render_pixels(
     show_pixels(grid = c(28, 28), size = c(250, 250),
                brush = matrix(c(0.5,1,1,0.5), 2, 2))
    )
  })
}
