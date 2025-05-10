library(shiny)
library(httr)
library(jsonlite)
library(DT)

ui <- fluidPage(
  titlePanel("Python with FastAPI & Pycaret"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("dataset", "Load a dataset", accept = c(".csv")),
      selectInput("analysis_type", "Type of analysis",
                  choices = c("AutoML", "Supervised learning", "Unsupervised learning")),
      
      conditionalPanel(
        condition = "input.analysis_type == 'Supervised learning'",
        uiOutput("target_ui"),
        uiOutput("features_ui"),
        selectInput("model_type", "Modèle",
                    choices = c("logistic_regression", "random_forest", "decision_tree", "svm")),
        sliderInput("test_split", "Train/Test split", min = 0.1, max = 0.9, value = 0.2),
        actionButton("train_btn", "Entraîner le modèle"),
        #verbatimTextOutput("training_output"),
        fileInput("predict_file", "Import a dataset for prediction"),
        selectInput("model_name", "Choose saved model :", choices = NULL),
        actionButton("predict_btn", "Do prediction"),
        #verbatimTextOutput("predict_result")
      ),
      conditionalPanel(
      condition = "input.analysis_type == 'AutoML'",
      uiOutput("automl_target_ui"),
      actionButton("automl_btn", "Lauch AutoML"),
      h4("AutoML Results "),
      #tableOutput("automl_leaderboard"),
      verbatimTextOutput("automl_best_model")
    )
,
      
      conditionalPanel(
        condition = "input.analysis_type == 'Unsupervised learning'",
        selectInput("unsup_model_type", "Modèle", choices = c("kmeans", "pca")),
        numericInput("n_clusters", "Number of clusters (KMeans)", value = 3, min = 2),
        numericInput("n_components", "Number of components (PCA)", value = 2, min = 2, max = 5),
        actionButton("train_unsup_btn", "Lauch unsupervised model"),
        #verbatimTextOutput("unsup_metrics"),
        #plotOutput("cluster_plot")
      )
    ),
    
mainPanel(
  h4("Output Analysis"),
  

  # Résultats AutoML
  conditionalPanel(
    condition = "input.analysis_type == 'AutoML'",
    h4("AutoML Results"),
    #tableOutput("automl_leaderboard"),
    DT::dataTableOutput("automl_leaderboard"),
    verbatimTextOutput("automl_best_model")
  ),

  # Résultats apprentissage supervisé
  conditionalPanel(
    condition = "input.analysis_type == 'Supervised learning'",
    h4("Rapport d'entraînement"),
    verbatimTextOutput("training_output"),
    h4("Prediction results"),
    verbatimTextOutput("predict_result")
  ),

  # Résultats apprentissage non supervisé
  conditionalPanel(
    condition = "input.analysis_type == 'Unsupervised learning'",
    h4("Métriques du modèle non supervisé"),
    verbatimTextOutput("unsup_metrics"),
    plotOutput("cluster_plot")
  ),

  hr(),
  h4("Overview of dataset"),
  dataTableOutput("preview_data")
)

  )
)

server <- function(input, output, session) {
  data <- reactiveVal(NULL)
  
  observeEvent(input$dataset, {
    req(input$dataset)
    df <- read.csv(input$dataset$datapath)
    data(df)
    updateSelectInput(session, "target_col", choices = names(df))
    output$preview_data <- renderDataTable({ df })
  })

    output$automl_target_ui <- renderUI({
      req(data())
      selectInput("automl_target", "Variable cible (AutoML)", choices = names(data()))
    })

    observeEvent(input$automl_btn, {
      req(input$automl_target)

      res <- POST(
        url = "http://127.0.0.1:8000/automl",
        body = list(
          file = upload_file(input$dataset$datapath),
          target = input$automl_target
        ),
        encode = "multipart"
      )

      if (res$status_code == 200) {
        result <- content(res, "parsed")
        output$automl_leaderboard <- DT::renderDataTable({
          req(result$leaderboard)
          DT::datatable(as.data.frame(result$leaderboard), options = list(pageLength = 10))
        })
        output$automl_best_model <- renderPrint({
          paste("🏆 Best model :", result$best_model_name)
        })
      } else {
        output$automl_best_model <- renderPrint({
          paste("❌ Erreur AutoML:", res$status_code, "\n", content(res, "text"))
        })
      }
    })


  output$target_ui <- renderUI({
    req(data())
    selectInput("target_col", "Variable cible", choices = names(data()))
  })

  output$features_ui <- renderUI({
    req(data())
    selectInput("features", "Variables explicatives", choices = names(data()), multiple = TRUE)
  })

  # Entraînement supervisé
  observeEvent(input$train_btn, {
    req(input$target_col, input$features, input$model_type)
    file_path <- input$dataset$datapath
    
    train_resp <- POST(
      url = "http://127.0.0.1:8000/train_sklearn",
      body = list(
        file = upload_file(file_path),
        target = input$target_col,
        features = paste(input$features, collapse = ","),
        model_type = input$model_type,
        test_size = as.character(input$test_split)
      ),
      encode = "multipart"
    )
    
    if (train_resp$status_code == 200) {
      result <- content(train_resp, "parsed")
      updateSelectInput(session, "model_name", choices = result$model_name)
      output$training_output <- renderPrint({ result$report })
    } else {
      output$training_output <- renderPrint({
        cat("❌ Erreur d'entraînement :", train_resp$status_code, "\n")
        print(content(train_resp, "text"))
      })
    }
  })

  # Prédiction
  observeEvent(input$predict_btn, {
    req(input$predict_file, input$model_name)
    res <- POST(
      url = "http://127.0.0.1:8000/predict-custom",
      body = list(
        file = upload_file(input$predict_file$datapath),
        model_name = input$model_name
      ),
      encode = "multipart"
    )
    
    output$predict_result <- renderPrint({
      if (res$status_code == 200) {
        jsonlite::fromJSON(content(res, "text"))
      } else {
        paste("❌ Erreur:", res$status_code, "\n", content(res, "text"))
      }
    })
  })

  # Modèles non supervisés
  observeEvent(input$train_unsup_btn, {
    req(input$dataset)
    
    file_path <- input$dataset$datapath
    
    unsup_resp <- POST(
      url = "http://127.0.0.1:8000/run_unsupervised",
      body = list(
        file = upload_file(file_path),
        model_type = input$unsup_model_type,
        n_clusters = as.character(input$n_clusters),
        n_components = as.character(input$n_components)
      ),
      encode = "multipart"
    )
    
    if (unsup_resp$status_code == 200) {
      result <- content(unsup_resp, "parsed")
      output$unsup_metrics <- renderPrint({ result$metrics })
      
      x_vals <- as.numeric(unlist(result$plot_data$x))
      y_vals <- as.numeric(unlist(result$plot_data$y))
      labels <- as.factor(unlist(result$plot_data$labels))

      if (any(is.na(x_vals)) || any(is.na(y_vals))) {
        output$cluster_plot <- renderPlot({
          plot(1, type = "n", main = "Erreur : données invalides")
        })
      } else {
        df <- data.frame(x = x_vals, y = y_vals, cluster = labels)
        output$cluster_plot <- renderPlot({
          plot(df$x, df$y, col = df$cluster, pch = 19,
               xlab = "Composante 1", ylab = "Composante 2",
               main = paste("Visualisation des clusters -", input$unsup_model_type))
          legend("topright", legend = levels(df$cluster),
                 col = 1:length(levels(df$cluster)), pch = 19)
        })
      }
    } else {
      output$unsup_metrics <- renderPrint({
        cat("❌ Erreur d'entraînement non supervisé :", unsup_resp$status_code, "\n")
        print(content(unsup_resp, "text"))
      })
    }
  })
}

shinyApp(ui, server)
