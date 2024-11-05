#
#   File:           pdfExtract.py
#   Description:    Functions that allow for the extraction and parsing
#                   of a project manual pdf.
#



#   Includes
#----------------------------------------------------------------------#
from PyPDF2 import PdfReader
import pymupdf as pdf

#**********************************************************************#



#   Function:   get_pdf_location
#   Parameters: none
#   Returns:    file_path (string | relative path to file)
#----------------------------------------------------------------------#
def get_pdf_location():
    """Function to get input from the user"""
    file_path = input("Please enter the relative pdf location: ")
    return file_path
#**********************************************************************#



#   Function:   extractPDFText
#   Parameters: pdf_file_path (string)
#   Returns:    pages (array | text_by_page[i] = text on page i)
#----------------------------------------------------------------------#
def extractPDFText(pdfFP):
    doc, pages = pdf.open(pdfFP), []        #Open pdf as document
    
    for i in range(doc.page_count):
        pages += [doc.get_page_text(i)]     #extract each page as text
        
    return pages
#**********************************************************************#



#   Function:   getTOC
#   Parameters: doc (custom object -- document)
#   Returns:    toc (array | array | toc[i][0] = lvl, toc[i][1] = title
#                                    toc[i][2] = start, toc[i][3] = end)
#----------------------------------------------------------------------#
def getTOC(doc):
    toc = doc.get_toc()             #Get table of contents
    
    #Add ending page numbers to each
    [toc[i].append(toc[i + 1][2]) for i in range(len(toc) - 1)]
    toc[-1].append(doc.page_count)  #Add last page
    
    return toc




#   python3 -c "import pdfExtract; print(pdfExtract.test())"
#   Testing function(s):
def printTest(text):
    for _ in text:
        print(_, '\n\n')

def test():
    # filePath = get_pdf_location()
    doc = pdf.open("project_manual.pdf")
    
    printTest(getTOC(doc))
    
    
    
    # text = extract_text_from_pdf("project_manual.pdf")
    # parseTableOfContents(text)
    # print(text[1])
    
    
    
    return "\nTest Done\n"
    
    