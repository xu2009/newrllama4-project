\name{get_lib_path}
\alias{get_lib_path}
\title{Get Backend Library Path}
\description{
Returns the full file path to the installed backend library.
}
\usage{
get_lib_path()
}
\value{
Character string containing the full path to the backend library file.
}
\details{
This function requires that the backend library has already been installed.
If the library is not installed, an error will be thrown.
}
\examples{
\dontrun{
# Get library path (requires library to be installed)
path <- get_lib_path()
print(path)
}
}
\seealso{
\code{\link{install_newrllama}}, \code{\link{lib_is_installed}}
} 