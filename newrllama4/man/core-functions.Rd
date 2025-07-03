\name{core-functions}
\alias{backend_init}
\alias{backend_free}
\alias{model_load}
\alias{context_create}
\alias{tokenize}
\alias{detokenize}
\alias{apply_chat_template}
\alias{generate}
\alias{generate_parallel}
\alias{tokenize_test}
\title{Core newrllama4 Functions}
\description{
Core functions for working with large language models through the llama.cpp backend.
}
\usage{
backend_init()
backend_free()
model_load(model_path, n_gpu_layers = 0L, use_mmap = TRUE, use_mlock = FALSE)
context_create(model, n_ctx = 2048L, n_threads = 4L, n_seq_max = 1L)
tokenize(model, text, add_special = TRUE)
detokenize(model, tokens)
apply_chat_template(model, messages, template = NULL, add_assistant = TRUE)
generate(context, tokens, max_tokens = 100L, top_k = 40L, top_p = 0.9, 
         temperature = 0.8, repeat_last_n = 64L, penalty_repeat = 1.1, 
         seed = -1L)
generate_parallel(context, prompts, max_tokens = 100L, top_k = 40L, 
                  top_p = 0.9, temperature = 0.8, repeat_last_n = 64L, 
                  penalty_repeat = 1.1, seed = -1L)
tokenize_test(model)
}
\arguments{
\item{model_path}{Path to the GGUF model file}
\item{n_gpu_layers}{Number of layers to offload to GPU (default: 0)}
\item{use_mmap}{Whether to use memory mapping (default: TRUE)}
\item{use_mlock}{Whether to use memory locking (default: FALSE)}
\item{model}{A model object returned by model_load()}
\item{n_ctx}{Context size (default: 2048)}
\item{n_threads}{Number of threads (default: 4)}
\item{n_seq_max}{Maximum number of sequences (default: 1)}
\item{text}{Text to tokenize}
\item{add_special}{Whether to add special tokens (default: TRUE)}
\item{tokens}{Integer vector of token IDs}
\item{messages}{List of chat messages, each with 'role' and 'content'}
\item{template}{Optional custom template (default: NULL, use model's template)}
\item{add_assistant}{Whether to add assistant prompt (default: TRUE)}
\item{context}{A context object returned by context_create()}
\item{prompts}{Character vector of prompts}
\item{max_tokens}{Maximum tokens to generate (default: 100)}
\item{top_k}{Top-k sampling (default: 40)}
\item{top_p}{Top-p sampling (default: 0.9)}
\item{temperature}{Sampling temperature (default: 0.8)}
\item{repeat_last_n}{Repetition penalty last n tokens (default: 64)}
\item{penalty_repeat}{Repetition penalty strength (default: 1.1)}
\item{seed}{Random seed (default: -1 for random)}
}
\value{
Functions return different types depending on their purpose:
\itemize{
  \item \code{model_load} returns a model object (external pointer)
  \item \code{context_create} returns a context object (external pointer)
  \item \code{tokenize} returns an integer vector of token IDs
  \item \code{detokenize} returns a character string
  \item \code{apply_chat_template} returns a formatted prompt string
  \item \code{generate} returns generated text
  \item \code{generate_parallel} returns a character vector of generated texts
  \item \code{tokenize_test} returns an integer vector of tokens for "H"
}
}
\details{
These functions provide the core interface to the llama.cpp backend library.
The backend library must be installed using \code{install_newrllama()} before 
these functions can be used.

Basic workflow:
1. Load a model with \code{model_load()}
2. Create a context with \code{context_create()}  
3. Use \code{tokenize()}, \code{generate()}, etc. for inference
}
\examples{
\dontrun{
# First install the backend
install_newrllama()

# Load a model
model <- model_load("path/to/model.gguf")

# Create context
context <- context_create(model)

# Tokenize text
tokens <- tokenize(model, "Hello world")

# Generate text  
result <- generate(context, tokens)
}
}
\seealso{
\code{\link{install_newrllama}}, \code{\link{newrllama4-package}}
} 