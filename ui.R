library(shiny)
library(keras)
library(pixels)
library(reticulate)




options(shiny.maxRequestSize=50*1024^2) 
ui <- fluidPage(
  titlePanel(title=div(img(src="logo.png",align='right'),"Predict Handwritten Digit")),
  
  sidebarPanel(
    
  
    fileInput("file", "Upload Trained Model")
    
   # submitButton(text = "Upload Model", icon("upload"))
  ),
  mainPanel( 
    
    tabsetPanel(type = "tabs",
                tabPanel("Overview",
                         br(),
                         p("Predict Handwritten Digit app is based on CNN (Convolutional Neural Network) model. 
                           There are following two tabs:"),
                         br(),
                         p("1. Model Development: This tab explains the process of model develoment"),
                          br(),
                         p("2. Data Input and Prediction: We are providing you trained model to upload and predict the handwritten digit.",
                           align="justify"),
                         #a(href="https://en.wikipedia.org/wiki/Market_segmentation","- Wikipedia"),
                         img(src = "MNIST.gif", height = 400, width = 500),
                         p("Model Summary:"),
                         verbatimTextOutput('model_summary'),
                         br()),
                tabPanel("Model Development",includeMarkdown('model_training.md')),
                tabPanel("Data Input and Prediction",
                         h4(p("Draw single digit number")),
                         tags$style(HTML("
                            #pixels {
                              height: 270px !important;
                              margin-top: 10px;
                            }
                          ")),
                         shiny_pixels_output("pixels"),
                         h5("click twice"),
                         actionButton("captureDigit", "Refresh",icon('refresh')),
                         
                         #actionButton("captureDigit", "Predict",icon('fingerprint')),
                         br(),
                        # p("Below is the predicted number"),
                         textOutput('predicted_number'))
                 
                        
                         
    )
  )
                
)
