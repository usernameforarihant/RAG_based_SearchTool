# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException, NoSuchElementException
# import time

# class AnalyticsVidhyaScraper:
#     def __init__(self):
#         options = webdriver.ChromeOptions()
#         # Add options to make browser more stable
#         options.add_argument('--ignore-certificate-errors')
#         options.add_argument('--ignore-ssl-errors')
#         self.driver = webdriver.Chrome(options=options)
#         self.wait = WebDriverWait(self.driver, 10)
#         self.base_url = "https://courses.analyticsvidhya.com/pages/all-free-courses"
    
#     def get_course_links(self):
#         course_links = []
        
#         # Handle pagination (9 pages)
#         for page in range(1, 10):
#             try:
#                 if page > 1:
#                     # Construct pagination URL
#                     url = f"{self.base_url}?page={page}"
#                     self.driver.get(url)
#                     time.sleep(2)
#                 else:
#                     self.driver.get(self.base_url)
                
#                 # Find all course cards
#                 course_cards = self.wait.until(
#                     EC.presence_of_all_elements_located((By.CLASS_NAME, "course-card__body"))
#                 )

#                 # print(course_cards.)
                
#                 # Extract links from each card
#                 for card in course_cards:
#                     try:
#                         link = card.find_element(By.TAG_NAME, "a").get_attribute("href")
#                         print
#                         course_links.append(link)
#                     except NoSuchElementException:
#                         continue
                
#                 print(f"Collected links from page {page}")
                
#             except Exception as e:
#                 print(f"Error processing page {page}: {str(e)}")
#                 continue
                
#         return course_links
    
#     def scrape_course_details(self, course_links):
#         courses_data = []
        
#         for link in course_links:
#             try:
#                 self.driver.get(link)
#                 time.sleep(2)  # Wait for content to load
                
#                 # Get course title
#                 title = self.wait.until(
#                     EC.presence_of_element_located((By.CSS_SELECTOR, "h1.course-title"))
#                 ).text
                
#                 # Get description
#                 try:
#                     description = self.driver.find_element(
#                         By.CSS_SELECTOR, "div.course-description"
#                     ).text
#                 except NoSuchElementException:
#                     description = "Description not available"
                
#                 # Get curriculum
#                 try:
#                     # Click on curriculum tab if it exists
#                     curriculum_tab = self.driver.find_element(
#                         By.CSS_SELECTOR, "[data-tab='curriculum']"
#                     )
#                     curriculum_tab.click()
#                     time.sleep(1)
                    
#                     curriculum_items = self.driver.find_elements(
#                         By.CSS_SELECTOR, "div.curriculum-item"
#                     )
#                     curriculum = "\n".join([item.text for item in curriculum_items])
#                 except NoSuchElementException:
#                     curriculum = "Curriculum not available"
                
#                 courses_data.append({
#                     'title': title.strip(),
#                     'description': description.strip(),
#                     'curriculum': curriculum.strip().replace('\n', ' | ')
#                 })
                
#                 print(f"Scraped: {title}")
                
#             except Exception as e:
#                 print(f"Error processing course {link}: {str(e)}")
#                 continue
        
#         return courses_data
    
#     def save_to_file(self, courses_data, filename="av_courses.txt"):
#         try:
#             with open(filename, 'w', encoding='utf-8') as f:
#                 for course in courses_data:
#                     f.write(f"title: {course['title']}/description: {course['description']}/curriculum: {course['curriculum']}\n")
#             print(f"Data successfully saved to {filename}")
#         except Exception as e:
#             print(f"Error saving to file: {str(e)}")
    
#     def run(self):
#         try:
#             print("Starting to collect course links...")
#             course_links = self.get_course_links()
#             print(f"Found {len(course_links)} courses")
            
#             print("Starting to scrape course details...")
#             courses_data = self.scrape_course_details(course_links)
            
#             self.save_to_file(courses_data)
            
#         except Exception as e:
#             print(f"Error during scraping: {str(e)}")
        
#         finally:
#             self.driver.quit()


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time

def get_course_links():
    # Setup Chrome driver with options
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')
    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 10)
    
    course_links = []
    base_url = "https://courses.analyticsvidhya.com/pages/all-free-courses"
    
    try:
        # Loop through all 9 pages
        for page in range(1, 10):
            try:
                # Construct URL with page number
                url = f"{base_url}?page={page}" if page > 1 else base_url
                driver.get(url)
                time.sleep(2)  # Wait for page to load
                
                # Find all course list items
                course_items = wait.until(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "li.course-cards__list-item")
                    )
                )
                
                # Extract links from each item
                for item in course_items:
                    try:
                        # Find the anchor tag within the list item
                        link = item.find_element(
                            By.CSS_SELECTOR, 
                            "a.course-card"
                        ).get_attribute("href")
                        
                        # Get the course title for verification
                        title = item.find_element(
                            By.CSS_SELECTOR, 
                            "h3"
                        ).text
                        
                        course_links.append({
                            'url': link,
                            'title': title
                        })
                        
                        print(f"Found course: {title}")
                        
                    except Exception as e:
                        print(f"Error extracting link from item: {str(e)}")
                        continue
                
                print(f"Completed page {page}")
                
            except Exception as e:
                print(f"Error processing page {page}: {str(e)}")
                continue
            
    except Exception as e:
        print(f"Error during scraping: {str(e)}")
        
    finally:
        driver.quit()
        
    return course_links

def save_links_to_file(course_links, filename="course_links.txt"):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for course in course_links:
                f.write(f"{course['url']}\n")
        print(f"Links saved to {filename}")
    except Exception as e:
        print(f"Error saving to file: {str(e)}")

def main():
    print("Starting to collect course links...")
    course_links = get_course_links()
    print(f"\nFound total of {len(course_links)} courses")
    
    # Save links to file
    save_links_to_file(course_links)
    
    # Also print links to console
    print("\nCourse Links:")
    for course in course_links:
        print(f"{course['title']}: {course['url']}")

if __name__ == "__main__":
    main()
# if __name__ == "__main__":
#     scraper = AnalyticsVidhyaScraper()
#     scraper.run()