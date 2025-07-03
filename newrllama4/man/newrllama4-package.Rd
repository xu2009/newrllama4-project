\name{newrllama4-package}
\alias{newrllama4-package}
\alias{newrllama4}
\docType{package}
\title{R Interface to llama.cpp with Runtime Library Loading}
\description{
Provides R bindings to the llama.cpp library for running large language models.
This package uses a lightweight architecture where the C++ backend library is downloaded
at runtime rather than bundled with the package.
}
\details{
After installation, users must run \code{install_newrllama()} to download the appropriate 
pre-compiled backend library for their system. The package supports text generation, 
tokenization, and chat template functionality.

Key functions:
\itemize{
  \item \code{\link{install_newrllama}} - Install backend library
  \item \code{\link{model_load}} - Load a language model
  \item \code{\link{generate}} - Generate text
  \item \code{\link{tokenize}} - Convert text to tokens
  \item \code{\link{detokenize}} - Convert tokens to text
}
}
\author{
yaoshengleo Developer <yaoshengleo@example.com>
}
\references{
\url{https://github.com/xu2009/newrllama4}
}
\keyword{package} 