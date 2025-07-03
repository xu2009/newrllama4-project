\name{install_newrllama}
\alias{install_newrllama}
\title{Install newrllama Backend Library}
\description{
Downloads and installs the pre-compiled C++ backend library required for the newrllama4 package to function.
}
\usage{
install_newrllama()
}
\details{
This function downloads platform-specific pre-compiled binaries from GitHub releases.
The backend library is stored in the user's data directory and loaded at runtime.
Internet connection is required for the initial download.
}
\value{
Returns NULL invisibly. Called for side effects.
}
\examples{
\dontrun{
# Install the backend library
install_newrllama()
}
}
\seealso{
\code{\link{lib_is_installed}}, \code{\link{get_lib_path}}
} 