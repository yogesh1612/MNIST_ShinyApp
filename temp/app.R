library(shiny)

ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      #pixels {
        height: 270px !important;
        margin-top: 10px;
      }
    "))
  ),
  titlePanel("Digit Capture Application"),
  #textOutput("prompt"),
  shiny_pixels_output("pixels"),
  actionButton("captureDigit", "Capture")
)








server <- function(input, output) {
  output$pixels <- shiny_render_pixels(
    show_pixels(grid = c(28, 28), size = c(250, 250),
                brush = matrix(c(0.5,1,1,0.5), 2, 2))  )
  
  #digit <- reactiveVal(floor(runif(1, 1, 10)))
 # output$prompt <- renderText(paste0("Please draw number ", digit(), ":"))
  
  observeEvent(input$captureDigit, {
    #digit_path <- file.path("digits", digit())
    #if (!dir.exists(digit_path)) dir.create(digit_path, recursive = TRUE)
    predictionss(input$pixels)
    
    #digit(floor(runif(1, 1, 10)))
    output$pixels <- shiny_render_pixels(
      show_pixels(grid = c(28, 28), size = c(250, 250),
                  brush = matrix(c(0.5,1,1,0.5), 2, 2))
    )
  })
}

shinyApp(ui = ui, server = server)
