% Generated manually; update with launch.R if the function changes.
\name{launch}
\alias{launch}
\title{Launch the classifier results explorer}
\usage{
launch(port = 8000L)
}
\arguments{
\item{port}{Localhost port for the explorer. Port 8000 is recommended because
it has been used for EcoSounds CORS testing.}
}
\value{
The server is returned invisibly after it has stopped.
}
\description{
Starts a local HTTP server with the cross-origin isolation headers required by
\code{oe-spectrogram}, opens the bundled explorer in the default browser, and
runs until interrupted.
}
