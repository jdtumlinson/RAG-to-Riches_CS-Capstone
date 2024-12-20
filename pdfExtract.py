#
#   File:           pdfExtract.py
#   Description:    Functions that allow for the extraction and parsing
#                   of a project manual pdf.
#



#   Includes
#----------------------------------------------------------------------#
import pymupdf as pdf
from dataclasses import dataclass
#**********************************************************************#



#----------------------------------------------------------------------#
@dataclass
class image:
    """Dataclass to store information on images in a pdf
    
        Fields:
            base_image (dict): A dictionary of information related to the image -- uses 'fitz' library
            page_loc (int): Page number of image
    """
    base_image: dict
    page_loc: int
#**********************************************************************#



#----------------------------------------------------------------------#
@dataclass
class section:
    """Dataclass to store information on sections
    
        Fields:
            title (str): Title of section
            text (str): All text on the pages of the section (may include other sections text)
            p_start (int): The start page of the section
            p_end (int): The end page of the section
            images (list): A list of all images within the section. Each item in the list is made of 'image' dataclass
    """
    title: str
    text: str
    p_start: int
    p_end: int
    images: list
#**********************************************************************#



#----------------------------------------------------------------------#
def extract_all(file_path: str, incl_images = False) -> tuple[list, list, list]:
    """Extracts text by page, a table of contents, and a list of images if specified

    Args:
        file_path (str): Relative path to file.
        incl_images (bool, optional): Set to True to include image information. Defaults to False.

    Returns:
        tuple[list, list, list]: pages, toc (returns None if cannot be formed), images
    """
    #Extract pages
    pages = extract_pages(file_path)
    
    #Extract table of contents
    toc = extract_toc(file_path)

    #Extract images
    images = extract_images(file_path) if incl_images else None

    return pages, toc, images
#**********************************************************************#



#----------------------------------------------------------------------#
def extract_pages(file_path: str) -> list:
    """Extracts text by page

    Args:
        file_path (str): Relative path to file.

    Returns:
        list
    """
    doc, pages = pdf.open(file_path), []
    
    for page in range(doc.page_count): 
        pages.append(doc.get_page_text(page))
    
    return pages
#**********************************************************************#



#----------------------------------------------------------------------#
def extract_toc(file_path: str) -> list:
    """Extracts a table of contents

    Args:
        file_path (str): Relative path to file.

    Returns:
        list
    """
    doc = pdf.open(file_path)
    toc = doc.get_toc()
    
    if toc != []: return create_toc(doc, bolded = False, font_size = True, fs_const = 0.1158)
    
    [toc[i].append(toc[i + 1][2]) for i in range(len(toc) - 1)]
    toc[-1].append(doc.page_count)
    
    return toc
#**********************************************************************#



#----------------------------------------------------------------------#
def create_toc(doc: pdf.Document, bolded = False, font_size = False, fs_const = 0.1) -> list:
    """Creates a table of contents based on bolded text and/or font size

    Args:
        doc (pdf.Document): Document to create TOC from
        bolded (bool, optional): Set to True to include all bolded text as the start of a section. Defaults to False.
        font_size (bool, optional): Set to True to include all text above a specific font size as the start of a section. Defaults to False.
        fs_const (float, optional): A constant value multiplied against the average font size to be added to the most common font size. Only applicable when 'font_size' is set to True. Defualts to 0.1.

    Returns:
        list
    """
    headers, min_fs = [], 0.0
    
    if font_size: min_fs = calc_fs_const(doc, fs_const)

    for page_i, page in enumerate(doc):
        blocks = page.get_text("dict", flags = 11)["blocks"]
        
        for b in blocks:
            for l in b["lines"]:
                for s in l["spans"]:
                    if bolded and s["flags"] == 20:
                        headers.append([s["text"], s["origin"][1], page_i + 1])
                    elif font_size and s["size"] > min_fs: 
                        headers.append([s["text"], s["origin"][1], page_i + 1])

    f_headers, y_pos, text, page_num = [], headers[0][1], "", 1
    for vals in headers:
        if vals[1] != y_pos:
            f_headers.append([text, page_num, vals[2]])
            
            text, y_pos, page_num = vals[0], vals[1], vals[2] 
        else:
            text += vals[0]
     
    return f_headers
#**********************************************************************#



#----------------------------------------------------------------------#
def calc_fs_const(doc: pdf.Document, fs_const: float) -> float:
    """Calculates a minimum font size where any size larger is labeled as a TOC section title\n
    Takes the most common font size of the first 3 pages or len(doc) // 3 + 1 (which ever is larger) and adds fs_const multiplied by the average font size over the same span

    Args:
        doc (pdf.Document): Document to calculate value based on
        fs_const (float): A constant value multiplied against the average font size to be added to the most common font size. Only applicable when 'font_size' is set to True. Defualts to 0.1.

    Returns:
        float
    """
    fontSizes, totalSpans, sumSize = {}, 0, 0.0

    for page in doc[ : max(len(doc) // 3 + 1, 3 if len(doc) > 3 else len(doc))]:
        blocks = page.get_text("dict", flags = 11)["blocks"]
        
        for b in blocks:
            for l in b["lines"]:
                for s in l["spans"]:
                    sumSize += s["size"]
                    totalSpans += 1
                    
                    if round(s["size"], 2) not in fontSizes: fontSizes[round(s["size"], 2)] = 1
                    else: fontSizes[round(s["size"], 2)] += 1
    
    modeKey, modeTally = 0, 0
    for size in fontSizes.keys():
        if fontSizes[size] > modeTally: modeKey = size
    
    constant = sumSize / totalSpans * fs_const
    
    return modeKey + constant
#**********************************************************************#



#----------------------------------------------------------------------#
def extract_images(file_path: str, start = 0, end = 0) -> list:
    """Returns a list of images and some of their metadata

    Args:
        file_path (str): Relative path to file.
        start (int, optional): Index of first page to extract images.
        end (int, optional): Index of last page to extract images.

    Returns:
        list
    """
    doc, images = pdf.open(file_path), []
    
    #Extract images
    for page_i in range(0 if end == 0 else start, len(doc) if end == 0 else end):        
        for _, img, in enumerate(doc[page_i].get_images()):
            images.append(image(doc.extract_image(img[0]), page_i))
    
    return images
#**********************************************************************#



#----------------------------------------------------------------------#
def extract_sections(file_path: str, incl_images = False) -> list:
    """Extracts a list of sections based on table of contents of file

    Args:
        file_path (str): Relative path to file.
        incl_images (bool, optional): Set to True to include image information. Defaults to False.

    Returns:
        list (returns None if cannot be formed)
    """
    toc, pages, sections = extract_toc(file_path), extract_pages(file_path), []
    
    if toc == None: return None
    
    for i in range(len(toc)):
        title = [toc[i][1]]
        text = "".join(pages[toc[i][2] - 1: toc[i][3]])
        images = []
        
        if incl_images: images = extract_images(file_path, start = toc[i][2] - 1, end = toc[i][3])
        else: images = None
        
        sections.append(section(title, text, toc[i][2], toc[i][3], images))
    
    return sections
#**********************************************************************#







#   python3 -c "import pdfExtract; print(pdfExtract.test())"
#   Testing function(s):
def printTest(text):
    for _ in text:
        print(_, '\n\n')

def test():
    # filePath = get_pdf_location()
    #doc = pdf.open()
    
    # text, toc, images = extract_all("pdf/5-20_262413_electrical_switchboard_response.pdf")
    # print(toc, "\n\n\n")
    
    text, toc, images = extract_all("pdf/testing_files/test.pdf", incl_images=True)
    print(toc)
    
    
    
    return "\nTest Done\n"
    
    