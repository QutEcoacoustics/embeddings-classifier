#' Launch the classifier results explorer
#'
#' Starts a localhost HTTP server, opens the bundled web interface, and blocks
#' until the user interrupts R.
#'
#' @param port Localhost port to use. EcoSounds currently requires an allowed
#'   origin; port 8000 is recommended.
#' @return The httpuv server, invisibly.
#' @export
launch <- function(port = 8000L) {
  root <- system.file("www", package = "resultsExplorer")
  if (!nzchar(root)) {
    stop("Could not find the packaged results explorer files.", call. = FALSE)
  }

  headers <- list(
    "Cross-Origin-Opener-Policy" = "same-origin",
    "Cross-Origin-Embedder-Policy" = "credentialless",
    "Cross-Origin-Resource-Policy" = "cross-origin"
  )

  server <- httpuv::startServer(
    "127.0.0.1",
    as.integer(port),
    list(
      staticPaths = list(
        "/" = httpuv::staticPath(root, headers = headers)
      )
    )
  )

  url <- sprintf("http://127.0.0.1:%d/", as.integer(port))
  message("Results explorer: ", url)
  utils::browseURL(url)

  tryCatch(
    repeat httpuv::service(1000L),
    finally = httpuv::stopServer(server)
  )

  invisible(server)
}
