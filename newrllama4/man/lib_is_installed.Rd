\name{lib_is_installed}
\alias{lib_is_installed}
\title{Check if Backend Library is Installed}
\description{
Checks whether the newrllama backend library has been downloaded and installed.
}
\usage{
lib_is_installed()
}
\value{
Logical value indicating whether the backend library is installed.
}
\examples{
# Check if backend library is installed
if (lib_is_installed()) {
  message("Backend library is ready")
} else {
  message("Please run install_newrllama() first")
}
}
\seealso{
\code{\link{install_newrllama}}, \code{\link{get_lib_path}}
} 