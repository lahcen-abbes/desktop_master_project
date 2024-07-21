from tkinter import *
import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from tkinter import Tk, Button, PhotoImage, Canvas
from tkinter import messagebox
import tkinter.font as font

def show_main_program():
    # Function to transition from landing page to main program
    landing_page.withdraw()  # Hide the landing page window
    root.deiconify()  # Show the main window
    root.state('zoomed')

# Create landing page window
landing_page = tk.Tk()
landing_page.title("Page d'atterrissage")
landing_page.state('zoomed')  

#You can easily set a logo for your Tkinter application by adding an icon to the top-left corner of the window. Here's how to do it:
# Load your logo image (replace 'path/to/your/logo.ico' with the actual path)
logo_image = PhotoImage(file='C:/Users/LAHCENE/OneDrive/Bureau/tkinter project/ABBES LOGO.png')

# Set the icon for both landing_page and root
landing_page.iconphoto(False, logo_image)

# Add image to the landing page
image_path = "C:/Users/LAHCENE/OneDrive/Bureau/tkinter project/2020002.png"
landing_image = PhotoImage(file=image_path)  # Assuming the image file is named 2020002.png
image_label = Label(landing_page, image=landing_image)
image_label.place(x=0, y=0, relwidth=1, relheight=1)
#image_label.pack(pady=20)

start_button = tk.Button(landing_page, text="Démarrer", command=show_main_program, 
                          font=("Arial", 12, "bold"),  
                          bg="#a13253",         
                          fg="#f0f0f0",             
                          activebackground="#f2bbcf", 
                          activeforeground="#a13253",    
                          padx=15, pady=7,          # Increase padding slightly 
                          relief="raised",           # Add a 3D raised effect
                          highlightthickness=2,     
                          highlightbackground="#a13253") 
                          

# Adjust rely to account for padding
start_button.place(relx=0.31, rely=0.75 - 9/landing_page.winfo_height(), anchor="center")

# Make the text bold and more white
start_button.config(width=21, height=3, bg="#a03253", fg="#f0f0f0", font=("Arial", 10, "bold")) 



#start_button.pack(pady=10)

# Create main window
root = tk.Toplevel()
root.state('zoomed')
root.configure(bg="#d56392")
root.title("Accueil")
root.withdraw()  # Hide the main window initially
root.iconphoto(False, logo_image)

# Add labels
#welcome_label = tk.Label(root, text="Bienvenue sur la page d'accueil!", font=("Helvetica", 16))
#welcome_label.pack(pady=20)

# Create a label to display the kernel type and accuracy
result_label = tk.Label(root, text="", font=("Helvetica", 12)) 
#result_label.pack(pady=10)   


# Function to count instances and attributes
def count_instances_and_attributes():
    # Read the CSV file
    csv_file_path = "C:/Users/LAHCENE/OneDrive/Bureau/tkinter project/data.csv"
    df = pd.read_csv(csv_file_path, sep=";")

    # Get the number of instances (lines) and attributes (columns)
    num_instances = len(df)
    num_attributes = len(df.columns)

    return num_instances, num_attributes

# Display the number of instances and attributes
num_instances, num_attributes = count_instances_and_attributes()

# --- Style the labels ---
label_font = ("Arial", 12)  
label_padding = (10, 5)  

instances_label = tk.Label(root, 
                          text=f"Nombre d'instances de la base de données (lignes): ",
                          font=label_font,
                          bg="#a03253", 
                          fg="#f0f0f0",
                          padx=label_padding[0], 
                          pady=label_padding[1])
instances_label.pack(pady=(20, 0))  # Add top padding 

# Create a separate label for num_instances with bold and bright styling
instances_value_label = tk.Label(root,
                                text=num_instances,
                                font=("Arial", 12, "bold"),
                                bg="#a03253",
                                fg="#f14d85")
instances_value_label.pack(pady=(0, 10))  # Add bottom padding

attributes_label = tk.Label(root, 
                            text=f"Nombre d'attributs de la base de données (colonnes): ",
                            font=label_font,
                            bg="#a13253",
                            fg="#f0f0f0",  
                            padx=label_padding[0],
                            pady=label_padding[1])
attributes_label.pack(pady=(20, 0))  # Add top padding

# Create a separate label for num_attributes with bold and bright styling
attributes_value_label = tk.Label(root,
                                  text=num_attributes,
                                  font=("Arial", 12, "bold"),
                                  bg="#a13253", 
                                  fg="#f14d85") 
attributes_value_label.pack(pady=(0, 10))  # Add bottom padding

def show_database_page():
    # Function to display the database page
    global database_window
    database_window = tk.Toplevel(root)
    database_window.state('zoomed')
    database_window.configure(bg="#d56392")
    database_window.iconphoto(False, logo_image)
    database_window.title("La Base De Données")
    
    # Button to return to the home page at the top of the database page
    return_button = tk.Button(database_window, 
                                 text="Retour à la page d'accueil", 
                                 command=return_to_home,
                                 font=("Arial", 11, "bold"),  
                                 bg="#D87093",         
                                 fg="#f0f0f0",             
                                 activebackground="#f0f0f0", 
                                 activeforeground="#D87093",    
                                 padx=15, pady=7,          # Increase padding slightly 
                                 relief="raised",           # Add a 3D raised effect
                                 highlightthickness=2,     
                                 highlightbackground="#a13253")
    return_button.pack(pady=(20,0))

    # Add content to the database page
    label = tk.Label(database_window, text="Voici Votre Base De Données",
                            font=label_font,
                            bg="#a13253",
                            fg="#f0f0f0",  
                            padx=label_padding[0],
                            pady=label_padding[1])
    label.pack(padx=20, pady=20)

    # Read the CSV file with semicolon as the separator
    csv_file_path = "C:/Users/LAHCENE/OneDrive/Bureau/tkinter project/data.csv"
    df = pd.read_csv(csv_file_path, sep=";")

    # Create a treeview widget for displaying the data
    style = ttk.Style()
    style.configure("Treeview.Heading", background="#ffebcc", font=("Arial", 10, "bold"))

    tree = ttk.Treeview(database_window, show="headings", style="Treeview")

    # Configure columns
    columns = df.columns.tolist()
    tree["columns"] = columns

    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)  # Set a fixed column width (adjust as needed)

    # Insert data into treeview
    for i, row in df.iterrows():
        tree.insert("", "end", text=i, values=row.tolist())

    # Add a vertical scrollbar
    vsb = ttk.Scrollbar(database_window, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")

    # Add a horizontal scrollbar
    hsb = ttk.Scrollbar(database_window, orient="horizontal", command=tree.xview)
    tree.configure(xscrollcommand=hsb.set)
    hsb.pack(side="bottom", fill="x")

    # Pack the treeview
    tree.pack(expand=True, fill="both")  # Allow treeview to expand

def return_to_home():
    # Function to return to the home page
    global database_window
    root.deiconify()  # Show the main window
    database_window.destroy()  # Destroy the database window

def poly_return_to_home():
    # Function to return to the home page
    global polynomial_window
    root.deiconify()  # Show the main window
    polynomial_window.destroy()

def rbf_return_to_home():
    # Function to return to the home page
    global gaussian_window
    root.deiconify()  # Show the main window
    gaussian_window.destroy()

def sig_return_to_home():
    # Function to return to the home page
    global sigmoid_params_window
    root.deiconify()  # Show the main window
    sigmoid_params_window.destroy()

def allk_return_to_home():
    # Function to return to the home page
    global all_kernel_params_window
    root.deiconify()  # Show the main window
    all_kernel_params_window.destroy()

def kres_return_to_home():
    # Function to return to the home page
    global kernel_window
    root.deiconify()  # Show the main window
    kernel_window.destroy()

def ts_return_to_home():
    # Function to return to the home page
    global test_size_window
    root.deiconify()  # Show the main window
    test_size_window.destroy()

def lin_return_to_home():
    # Function to return to the home page
    global test_size_window
    root.deiconify()  # Show the main window
    linear_window.destroy()

def return_to_landing():
    # Function to return to the home page
    global root
    landing_page.deiconify()  # Show the main window
    landing_page.state('zoomed')
    root.withdraw()  # Hide the database window

def create_database_button():
    # Function to create the button to show the database page
    database_button = tk.Button(root, 
                                 text="La Base De Données", 
                                 command=show_database_page,
                                 font=("Arial", 11, "bold"),  
                                 bg="#008080",         
                                 fg="#f0f0f0",             
                                 activebackground="#f0f0f0", 
                                 activeforeground="#008080",    
                                 padx=15, pady=7,          # Increase padding slightly 
                                 relief="raised",           # Add a 3D raised effect
                                 highlightthickness=2,     
                                 highlightbackground="#a13253")
    #database_button.place(x=300, y=100, width=200, height=50) 
    database_button.pack(pady=10)
    


global test_size_entry,test_size,test 
test_size_entry = None
test_size_window = None  # Define test_size_window here
test_size = 0.2  # Set a default test size here

def get_test_size(test):
    global test_size_entry
    try:
        print("la valeur de test_size_entry dans get test size : ",test_size_entry.get())
        test=test_size_entry.get()
        test = float(test_size_entry.get())
        print ("la valeur de test size :",test)
        if 0.1 <= test <= 0.9:
            pass
            #return test_size
        else:
            tk.messagebox.showerror("Erreur", "Veuillez entrer une valeur entre 0.1 et 0.9.")
            test_size_entry.delete(0, tk.END)
            test_size_entry.insert(0, "0.2")
            test=0.2
            #return 0.2
    except ValueError:
        tk.messagebox.showerror("Erreur", "Veuillez entrer un nombre valide.")
        test_size_entry.delete(0, tk.END)
        test_size_entry.insert(0, "0.2")
        test=0.2
        #return 0.2
    return test

# Initialize test_size_entry
test_size_entry = tk.Entry()

# Later in the code, assign it the entry widget created in show_test_size_input() function
#global test_size_entry
test_size_entry = tk.Entry(root)  # Create the entry widget within the main window
test_size_entry.pack_forget()  # Initially hide the entry

# Modify show_test_size_input:
def show_test_size_input():
    global test_size_window
    test_size_window = tk.Toplevel(root)
    test_size_window.state('zoomed')
    test_size_window.configure(bg="#d56392")
    test_size_window.iconphoto(False, logo_image)
    global test_size_entry
    test_size_window.title("La Taille Du Test")
    
    test_size_label = tk.Label(test_size_window, 
                               text="Entrez la taille du test (entre 0.1 et 0.9, par exemple, 0.2 pour 20%):",
                               font=label_font,
                               bg="#a03253", 
                               fg="#f0f0f0",
                               padx=label_padding[0], 
                               pady=label_padding[1])
    test_size_label.pack(pady=(30, 20))
    
    test_size_entry = tk.Entry(test_size_window, 
                               font=("Arial", 11),             
                               bg="#f2f2f2",                    
                               fg="#333333",                    
                               relief="solid",                   
                               borderwidth=1,                   
                               highlightthickness=0,            
                               insertbackground="#a03253")       
    test_size_entry.pack(pady=(0, 20))
    
    confirm_button = tk.Button(test_size_window, 
                               text="Confirmer", 
                               command=validate_test_size,
                               font=("Arial", 11, "bold"),     
                               bg="#28a745",                  
                               fg="#ffffff",                  
                               activebackground="#218838",   
                               activeforeground="#ffffff",   
                               padx=15, pady=8,               
                               relief="raised",                
                               highlightthickness=0)
    confirm_button.pack(pady=(10, 20))

    return_button = tk.Button(test_size_window, 
                              text="Retour à la page d'accueil", 
                              command=ts_return_to_home,
                              font=("Arial", 10, "bold"),  
                              bg="#D87093",         
                              fg="#f0f0f0",             
                              activebackground="#f0f0f0", 
                              activeforeground="#D87093",    
                              padx=10, pady=7,          
                              relief="raised",           
                              highlightthickness=2,     
                              highlightbackground="#a13253")
    return_button.pack(pady=(10, 30))



# Now you can safely call the get() method on test_size_entry
#test_size_value = test_size_entry.get()



def validate_test_size():
    global test_size_window
    global test_size,test
    test=0
    test_size=get_test_size(test)
    #test_size = get_test_size(test_size1)
    print("test size dans validate test size : ",test_size)
    if test_size:
        test_size_entry.delete(0, tk.END)
        test_size_entry.insert(0, str(test_size))
        test_size_window.destroy()



def show_kernel_results(kernel_type, accuracy, sensitivity, specificity, tn, fp, fn, tp, test_size_value, degree=None, coef=None, gamma=None, alpha=None, beta=None):
    global kernel_window
    kernel_window = tk.Toplevel(root)
    kernel_window.state('zoomed')
    kernel_window.configure(bg="#d56392")
    kernel_window.iconphoto(False, logo_image)
    kernel_window.title(f"Le Résultat du Noyau {kernel_type.capitalize()}") 


    result_label = tk.Label(kernel_window, text=f"Noyau: {kernel_type}, Précision: {accuracy * 100:.2f}%, Sensibilité: {sensitivity:.2f}, Spécificité: {specificity:.2f}, Vrai Négatif: {tn}, Faux Positif: {fp}, Faux Négatif: {fn}, Vrai Positif: {tp}, Taille du test: {test_size}",
                            font=font.Font(family="Helvetica", size=11, weight="bold"))
    result_label.pack(pady=20)

    accuracy_percentage = accuracy * 100

    if kernel_type == "poly" or kernel_type == "polynomial":
        result_label_poly = tk.Label(kernel_window, text=f"Noyau: Polynomial, Puissance d: {degree}, Coefficient c: {coef}, Précision: {accuracy_percentage:.2f}%, Taille du test: {test_size}",
                                     font=font.Font(family="Helvetica", size=11, weight="bold"))
        result_label_poly.pack(pady=10)

def handle_kernel(kernel_type, degree, coefficient_c, gamma=None, alpha=None, beta=None):
    data = pd.read_csv("C:/Users/LAHCENE/OneDrive/Bureau/tkinter project/data.csv", sep=";")
    y = data['diagnosis']
    X = data.drop('diagnosis', axis=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y, test_size=test_size, random_state=None)

    if kernel_type == "linear":
        svm_classifier = SVC(kernel='linear')
    elif kernel_type == "poly":
        svm_classifier = SVC(kernel='poly', degree=degree, coef0=coefficient_c)
    elif kernel_type == "gaussian":
        svm_classifier = SVC(kernel='rbf', gamma=gamma)
    elif kernel_type == "sigmoid":
        svm_classifier = SVC(kernel='sigmoid', coef0=alpha)

    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def show_all_kernel_parameters():
    global all_kernel_params_window
    all_kernel_params_window = tk.Toplevel(root)
    all_kernel_params_window.state('zoomed')
    all_kernel_params_window.configure(bg="#d56392")
    all_kernel_params_window.iconphoto(False, logo_image)
    all_kernel_params_window.title("Saisir Les Paramètres Et L'affichage Des Noyaux")


    degree_label = tk.Label(all_kernel_params_window, 
                               text="Puissance d du noyau polynomial:",
                               font=label_font,
                               bg="#a03253", 
                               fg="#f0f0f0",
                               padx=label_padding[0], 
                               pady=label_padding[1])
    degree_label.pack(pady=(20, 10))
    degree_entry = tk.Entry(all_kernel_params_window, 
                               font=("Arial", 11),             
                               bg="#f2f2f2",                    
                               fg="#333333",                    
                               relief="solid",                   
                               borderwidth=1,                   
                               highlightthickness=0,            
                               insertbackground="#a03253")       
    degree_entry.pack(pady=(0, 10))
    
    coef_label = tk.Label(all_kernel_params_window, 
                               text="Coefficient c du noyau polynomial:",
                               font=label_font,
                               bg="#a03253", 
                               fg="#f0f0f0",
                               padx=label_padding[0], 
                               pady=label_padding[1])
    coef_label.pack(pady=(0, 10))
    coef_entry = tk.Entry(all_kernel_params_window, 
                               font=("Arial", 11),             
                               bg="#f2f2f2",                    
                               fg="#333333",                    
                               relief="solid",                   
                               borderwidth=1,                   
                               highlightthickness=0,            
                               insertbackground="#a03253")       
    coef_entry.pack(pady=(0, 10))

    # Add input field for gamma parameter of Gaussian kernel
    gamma_label = tk.Label(all_kernel_params_window, 
                               text="Paramètre γ du noyau gaussien:",
                               font=label_font,
                               bg="#a03253", 
                               fg="#f0f0f0",
                               padx=label_padding[0], 
                               pady=label_padding[1])
    gamma_label.pack(pady=(0, 10))
    gamma_entry = tk.Entry(all_kernel_params_window, 
                               font=("Arial", 11),             
                               bg="#f2f2f2",                    
                               fg="#333333",                    
                               relief="solid",                   
                               borderwidth=1,                   
                               highlightthickness=0,            
                               insertbackground="#a03253")       
    gamma_entry.pack(pady=(0, 10))
    
    # Add input fields for Sigmoid kernel parameters
    alpha_label = tk.Label(all_kernel_params_window, 
                               text="Alpha du noyau Sigmoïde:",
                               font=label_font,
                               bg="#a03253", 
                               fg="#f0f0f0",
                               padx=label_padding[0], 
                               pady=label_padding[1])
    alpha_label.pack(pady=(0, 10))
    alpha_entry = tk.Entry(all_kernel_params_window, 
                               font=("Arial", 11),             
                               bg="#f2f2f2",                    
                               fg="#333333",                    
                               relief="solid",                   
                               borderwidth=1,                   
                               highlightthickness=0,            
                               insertbackground="#a03253")       
    alpha_entry.pack(pady=(0, 10))
    
    beta_label = tk.Label(all_kernel_params_window, 
                               text="Coefficient du noyau Sigmoïde:",
                               font=label_font,
                               bg="#a03253", 
                               fg="#f0f0f0",
                               padx=label_padding[0], 
                               pady=label_padding[1])
    beta_label.pack(pady=(0, 10))
    beta_entry = tk.Entry(all_kernel_params_window, 
                               font=("Arial", 11),             
                               bg="#f2f2f2",                    
                               fg="#333333",                    
                               relief="solid",                   
                               borderwidth=1,                   
                               highlightthickness=0,            
                               insertbackground="#a03253")       
    beta_entry.pack(pady=(0, 20))
    
    # Create a text widget to display the results
    results_text_widget = tk.Text(all_kernel_params_window, height=14, width=80)
    results_text_widget.pack(pady=10)
    
    calculate_button = tk.Button(all_kernel_params_window, 
                               text="Calculer les Results", 
                               command=lambda: show_all_kernel_results(degree_entry.get(), coef_entry.get(), gamma_entry.get(), alpha_entry.get(), beta_entry.get(), results_text_widget),
                               font=("Arial", 11, "bold"),     
                               bg="#28a745",                  
                               fg="#ffffff",                  
                               activebackground="#218838",   
                               activeforeground="#ffffff",   
                               padx=15, pady=8,               
                               relief="raised",                
                               highlightthickness=0)
    calculate_button.pack(pady=(10, 0))
    
    # Add a button to return to the home page
    return_button = tk.Button(all_kernel_params_window, 
                              text="Retour à la page d'accueil", 
                              command=allk_return_to_home,
                              font=("Arial", 10, "bold"),  
                              bg="#D87093",         
                              fg="#f0f0f0",             
                              activebackground="#f0f0f0", 
                              activeforeground="#D87093",    
                              padx=10, pady=7,          
                              relief="raised",           
                              highlightthickness=2,     
                              highlightbackground="#a13253")
    return_button.pack(pady=(10, 0))
    return_button.place(x=20, y=20)


def show_all_kernel_results(degree, coef, gamma, alpha, beta, results_text_widget):
    # Calculate and display results for all kernels

    # Linear kernel
    linear_accuracy = handle_kernel("linear", None, None, None, None, None)
    y_true, y_pred = get_true_pred("linear", None, None, None, None, None)  # Get true and predicted values
    linear_sensitivity, linear_specificity, linear_tn, linear_fp, linear_fn, linear_tp = calculate_sensitivity_specificity(y_true, y_pred)
    linear_result = f"Le Noyau Linéaire, Précision: {linear_accuracy * 100:.2f}%, sensibilité: {linear_sensitivity:.2f}, Spécificité: {linear_specificity:.2f}, Vrai Négatif: {linear_tn}, Faux Positif: {linear_fp}, Faux Négatif: {linear_fn}, Vrai Positif: {linear_tp}, Taille du test: {test_size}\n"
    
    # Polynomial kernel
    degree = int(degree)
    coef = float(coef)
    poly_accuracy = handle_kernel("poly", degree, coef, None, None, None)
    y_true, y_pred = get_true_pred("poly", degree, coef, None, None, None)  # Get true and predicted values
    poly_sensitivity, poly_specificity, poly_tn, poly_fp, poly_fn, poly_tp = calculate_sensitivity_specificity(y_true, y_pred)
    poly_result = f"\nLe Noyau Polynomial (Puissance d: {degree}, Coefficient c: {coef}): Précision: {poly_accuracy * 100:.2f}%, sensibilité: {poly_sensitivity:.2f}, Spécificité: {poly_specificity:.2f}, Vrai Négatif: {poly_tn}, Faux Positif: {poly_fp}, Faux Négatif: {poly_fn}, Vrai Positif: {poly_tp}, Taille du test: {test_size}\n"
    
    # Gaussian kernel
    gamma = float(gamma)
    gaussian_accuracy = handle_kernel("gaussian", None, None, gamma, None, None)
    y_true, y_pred = get_true_pred("gaussian", None, None, gamma, None, None)  # Get true and predicted values
    gaussian_sensitivity, gaussian_specificity, gaussian_tn, gaussian_fp, gaussian_fn, gaussian_tp = calculate_sensitivity_specificity(y_true, y_pred)
    gaussian_result = f"\nLe Noyau Gaussien (Gamma: {gamma}): Précision: {gaussian_accuracy * 100:.2f}%, sensibilité: {gaussian_sensitivity:.2f}, Spécificité: {gaussian_specificity:.2f}, Vrai Négatif: {gaussian_tn}, Faux Positif: {gaussian_fp}, Faux Négatif: {gaussian_fn}, Vrai Positif: {gaussian_tp}, Taille du test: {test_size}\n"
    
    # Sigmoid kernel
    alpha = float(alpha)
    beta = float(beta)
    sigmoid_accuracy = handle_kernel("sigmoid", None, None, None, alpha, beta)
    y_true, y_pred = get_true_pred("sigmoid", None, None, None, alpha, beta)  # Get true and predicted values
    sigmoid_sensitivity, sigmoid_specificity, sigmoid_tn, sigmoid_fp, sigmoid_fn, sigmoid_tp = calculate_sensitivity_specificity(y_true, y_pred)
    sigmoid_result = f"\nLe Noyau Sigmoïde (Alpha: {alpha}, Coefficient: {beta}): Précision: {sigmoid_accuracy * 100:.2f}%, sensibilité: {sigmoid_sensitivity:.2f}, Spécificité: {sigmoid_specificity:.2f}, Vrai Négatif: {sigmoid_tn}, Faux Positif: {sigmoid_fp}, Faux Négatif: {sigmoid_fn}, Vrai Positif: {sigmoid_tp}, Taille du test: {test_size}\n"
    
    # Create the font
    custom_font = font.Font(family="Helvetica", size=11, weight="bold")

    # Configure a text tag with the custom font
    results_text_widget.tag_configure("bold_font", font=custom_font)

    # Display results in the text widget
    results_text_widget.delete(1.0, tk.END)  # Clear existing content
    results_text_widget.insert(tk.END, linear_result, "bold_font")
    results_text_widget.insert(tk.END, poly_result, "bold_font")
    results_text_widget.insert(tk.END, gaussian_result, "bold_font")
    results_text_widget.insert(tk.END, sigmoid_result, "bold_font")




def get_true_pred(kernel_type, degree, coefficient_c, gamma, alpha, beta):
    data = pd.read_csv("C:/Users/LAHCENE/OneDrive/Bureau/tkinter project/data.csv", sep=";")
    y = data['diagnosis']
    X = data.drop('diagnosis', axis=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y, test_size=test_size, random_state=None)

    if kernel_type == "linear":
        svm_classifier = SVC(kernel='linear')
    elif kernel_type == "poly":
        svm_classifier = SVC(kernel='poly', degree=degree, coef0=coefficient_c)
    elif kernel_type == "gaussian":
        svm_classifier = SVC(kernel='rbf', gamma=gamma)
    elif kernel_type == "sigmoid":
        svm_classifier = SVC(kernel='sigmoid', coef0=alpha)

    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)

    return y_test, y_pred




def calculate_poly_results(degree, coef, results_text_widget):
    degree = int(degree)
    coef = float(coef)
    accuracy = handle_kernel("poly", degree, coef)
    y_true, y_pred = get_true_pred("poly", degree, coef, None, None, None)  # Get true and predicted values
    poly_sensitivity, poly_specificity, poly_tn, poly_fp, poly_fn, poly_tp = calculate_sensitivity_specificity(y_true, y_pred)
    
    # Clear the existing content in the text widget
    results_text_widget.delete(1.0, tk.END)
    
    accuracy_percentage = accuracy * 100
    t=test_size
    # Create the font
    custom_font = font.Font(family="Helvetica", size=11, weight="bold")

    # Configure a text tag with the custom font
    results_text_widget.tag_configure("bold_font", font=custom_font)

    # Insert text and apply the tag
    results_text_widget.insert(tk.END, "Noyau: Polynomial, Puissance d: ", "bold_font")
    results_text_widget.insert(tk.END, f"{degree}, ", "bold_font")
    results_text_widget.insert(tk.END, "Coefficient c: ", "bold_font")
    results_text_widget.insert(tk.END, f"{coef}, ", "bold_font")
    results_text_widget.insert(tk.END, "Précision: ", "bold_font")
    results_text_widget.insert(tk.END, f"{accuracy_percentage:.2f}%, ", "bold_font")
    results_text_widget.insert(tk.END, "Sensibilité: ", "bold_font")
    results_text_widget.insert(tk.END, f"{poly_sensitivity:.2f}, ", "bold_font")
    results_text_widget.insert(tk.END, "Spécificité: ", "bold_font")
    results_text_widget.insert(tk.END, f"{poly_specificity:.2f}, ", "bold_font")
    # Add similar lines for TN, FP, FN, TP
    results_text_widget.insert(tk.END, "Vrai Négatif: ", "bold_font")
    results_text_widget.insert(tk.END, f"{poly_tn}, ", "bold_font")
    results_text_widget.insert(tk.END, "Faux Positif: ", "bold_font")
    results_text_widget.insert(tk.END, f"{poly_fp}, ", "bold_font")
    results_text_widget.insert(tk.END, "Faux Négatif: ", "bold_font")
    results_text_widget.insert(tk.END, f"{poly_fn}, ", "bold_font")
    results_text_widget.insert(tk.END, "Vrai Positif: ", "bold_font")
    results_text_widget.insert(tk.END, f"{poly_tp}, ", "bold_font")
    results_text_widget.insert(tk.END, "Taille du test: ", "bold_font")
    results_text_widget.insert(tk.END, f"{test_size}\n", "bold_font")


def show_polynomial_parameters():
    # Function to display inputs required for polynomial kernel parameters
    global polynomial_window
    polynomial_window = tk.Toplevel(root)
    polynomial_window.state('zoomed')
    polynomial_window.configure(bg="#d56392")
    polynomial_window.iconphoto(False, logo_image)
    polynomial_window.title("Saisir Les Paramètres Et L'affichage Le Résultat Du Noyau Polynomial")


    degree_label = tk.Label(polynomial_window, 
                               text="Puissance d du noyau polynomial:",
                               font=label_font,
                               bg="#a03253", 
                               fg="#f0f0f0",
                               padx=label_padding[0], 
                               pady=label_padding[1])
    degree_label.pack(pady=(30, 20))
    degree_entry = tk.Entry(polynomial_window, 
                               font=("Arial", 11),             
                               bg="#f2f2f2",                    
                               fg="#333333",                    
                               relief="solid",                   
                               borderwidth=1,                   
                               highlightthickness=0,            
                               insertbackground="#a03253")       
    degree_entry.pack(pady=(0, 20))

    coef_label = tk.Label(polynomial_window, 
                               text="Coefficient c du noyau polynomial:",
                               font=label_font,
                               bg="#a03253", 
                               fg="#f0f0f0",
                               padx=label_padding[0], 
                               pady=label_padding[1])
    coef_label.pack(pady=(20, 20))
    coef_entry = tk.Entry(polynomial_window, 
                               font=("Arial", 11),             
                               bg="#f2f2f2",                    
                               fg="#333333",                    
                               relief="solid",                   
                               borderwidth=1,                   
                               highlightthickness=0,            
                               insertbackground="#a03253")       
    coef_entry.pack(pady=(0, 20))

    # Create a text widget to display the result
    results_text_widget = tk.Text(polynomial_window, height=5, width=60)
    results_text_widget.pack(pady=10)
    
    calculate_button = tk.Button(polynomial_window, 
                               text="Calculer Le Résultat", 
                               command=lambda: calculate_poly_results(degree_entry.get(), coef_entry.get(), results_text_widget),
                               font=("Arial", 11, "bold"),     
                               bg="#28a745",                  
                               fg="#ffffff",                  
                               activebackground="#218838",   
                               activeforeground="#ffffff",   
                               padx=15, pady=8,               
                               relief="raised",                
                               highlightthickness=0)
    calculate_button.pack(pady=(10, 20))

     # Add a button to return to the home page
    return_button = tk.Button(polynomial_window, 
                              text="Retour à la page d'accueil", 
                              command=poly_return_to_home,
                              font=("Arial", 10, "bold"),  
                              bg="#D87093",         
                              fg="#f0f0f0",             
                              activebackground="#f0f0f0", 
                              activeforeground="#D87093",    
                              padx=10, pady=7,          
                              relief="raised",           
                              highlightthickness=2,     
                              highlightbackground="#a13253")
    return_button.pack(pady=(10, 30))

    
def calculate_polynomial_results(degree, coef):
    # Function to calculate polynomial kernel results
    degree = int(degree)
    coef = float(coef)
    
    # Perform SVM with polynomial kernel
    accuracy = handle_kernel("poly", degree, coef)

    # Display the result in an independent page
    show_kernel_results("Polynomial", accuracy, degree=degree, coef=coef, test_size=test_size)
   

def calculate_gaussian_results(gamma, results_text_widget):
    # Gaussian kernel
    gamma = float(gamma)
    accuracy = handle_kernel("gaussian", None, None, gamma)
    y_true, y_pred = get_true_pred("gaussian", None, None, gamma, None, None)  # Get true and predicted values
    gaussian_sensitivity, gaussian_specificity, gaussian_tn, gaussian_fp, gaussian_fn, gaussian_tp = calculate_sensitivity_specificity(y_true, y_pred)

    # Display the result in the provided text widget
    results_text_widget.delete(1.0, tk.END)  # Clear existing content
    t=test_size

    # Create the font
    custom_font = font.Font(family="Helvetica", size=11, weight="bold")

    # Configure a text tag with the custom font
    results_text_widget.tag_configure("bold_font", font=custom_font)

    # Insert text and apply the tag
    results_text_widget.insert(tk.END, "Noyau: Gaussien, Gamma: ", "bold_font")
    results_text_widget.insert(tk.END, f"{gamma}, ", "bold_font")
    results_text_widget.insert(tk.END, "Précision: ", "bold_font")
    results_text_widget.insert(tk.END, f"{accuracy * 100:.2f}%, ", "bold_font")
    results_text_widget.insert(tk.END, "Sensibilité: ", "bold_font")
    results_text_widget.insert(tk.END, f"{gaussian_sensitivity:.2f}, ", "bold_font")
    results_text_widget.insert(tk.END, "Spécificité: ", "bold_font")
    results_text_widget.insert(tk.END, f"{gaussian_specificity:.2f}, ", "bold_font")
    # Add similar lines for TN, FP, FN, TP
    results_text_widget.insert(tk.END, "Vrai Négatif: ", "bold_font")
    results_text_widget.insert(tk.END, f"{gaussian_tn}, ", "bold_font")
    results_text_widget.insert(tk.END, "Faux Positif: ", "bold_font")
    results_text_widget.insert(tk.END, f"{gaussian_fp}, ", "bold_font")
    results_text_widget.insert(tk.END, "Faux Négatif: ", "bold_font")
    results_text_widget.insert(tk.END, f"{gaussian_fn}, ", "bold_font")
    results_text_widget.insert(tk.END, "Vrai Positif: ", "bold_font")
    results_text_widget.insert(tk.END, f"{gaussian_tp}, ", "bold_font")
    results_text_widget.insert(tk.END, "Taille du test: ", "bold_font")
    results_text_widget.insert(tk.END, f"{test_size}\n", "bold_font")


def show_gaussian_parameters():
    # Function to display inputs required for polynomial kernel parameters
    global gaussian_window
    gaussian_window = tk.Toplevel(root)
    gaussian_window.state('zoomed')
    gaussian_window.configure(bg="#d56392")
    gaussian_window.iconphoto(False, logo_image)
    gaussian_window.title("Saisir Les Paramètres Et L'affichage Le Résultat Du Noyau Gaussien")
    
    gamma_label = tk.Label(gaussian_window, 
                               text="Gamma du noyau gaussien:",
                               font=label_font,
                               bg="#a03253", 
                               fg="#f0f0f0",
                               padx=label_padding[0], 
                               pady=label_padding[1])
    gamma_label.pack(pady=(30, 20))
    gamma_entry = tk.Entry(gaussian_window, 
                               font=("Arial", 11),             
                               bg="#f2f2f2",                    
                               fg="#333333",                    
                               relief="solid",                   
                               borderwidth=1,                   
                               highlightthickness=0,            
                               insertbackground="#a03253")       
    gamma_entry.pack(pady=(0, 20))

    # Create a text widget to display the result
    results_text_widget = tk.Text(gaussian_window, height=5, width=65)
    results_text_widget.pack(pady=10)
    
    calculate_button = tk.Button(gaussian_window, 
                               text="Calculer Le Résultat", 
                               command=lambda: calculate_gaussian_results(gamma_entry.get(), results_text_widget),
                               font=("Arial", 11, "bold"),     
                               bg="#28a745",                  
                               fg="#ffffff",                  
                               activebackground="#218838",   
                               activeforeground="#ffffff",   
                               padx=15, pady=8,               
                               relief="raised",                
                               highlightthickness=0)
    calculate_button.pack(pady=(10, 20))
    
    # Add a button to return to the home page
    return_button = tk.Button(gaussian_window, 
                              text="Retour à la page d'accueil", 
                              command=rbf_return_to_home,
                              font=("Arial", 10, "bold"),  
                              bg="#D87093",         
                              fg="#f0f0f0",             
                              activebackground="#f0f0f0", 
                              activeforeground="#D87093",    
                              padx=10, pady=7,          
                              relief="raised",           
                              highlightthickness=2,     
                              highlightbackground="#a13253")
    return_button.pack(pady=(10, 30))
    
def show_sigmoid_params_window():
    # Function to display the window for setting the parameters α and β
    global sigmoid_params_window
    sigmoid_params_window = tk.Toplevel(root)
    sigmoid_params_window.state('zoomed')
    sigmoid_params_window.configure(bg="#d56392")
    sigmoid_params_window.iconphoto(False, logo_image)
    sigmoid_params_window.title("Saisir Les Paramètres Et L'affichage Le Résultat Du Noyau Sigmoïde")
    
    # Add labels and entries for α and β parameters
    alpha_label = tk.Label(sigmoid_params_window, 
                               text="Entrez le paramètre alpha (α):",
                               font=label_font,
                               bg="#a03253", 
                               fg="#f0f0f0",
                               padx=label_padding[0], 
                               pady=label_padding[1])
    alpha_label.pack(pady=(30, 20))
    alpha_entry = tk.Entry(sigmoid_params_window, 
                               font=("Arial", 11),             
                               bg="#f2f2f2",                    
                               fg="#333333",                    
                               relief="solid",                   
                               borderwidth=1,                   
                               highlightthickness=0,            
                               insertbackground="#a03253")       
    alpha_entry.pack(pady=(0, 20))
    
    beta_label = tk.Label(sigmoid_params_window, 
                               text="Entrez le paramètre Coefficient (c):",
                               font=label_font,
                               bg="#a03253", 
                               fg="#f0f0f0",
                               padx=label_padding[0], 
                               pady=label_padding[1])
    beta_label.pack(pady=(20, 20))
    beta_entry = tk.Entry(sigmoid_params_window, 
                               font=("Arial", 11),             
                               bg="#f2f2f2",                    
                               fg="#333333",                    
                               relief="solid",                   
                               borderwidth=1,                   
                               highlightthickness=0,            
                               insertbackground="#a03253")       
    beta_entry.pack(pady=(0, 20))
    beta_label = tk.Label(sigmoid_params_window, text="Entrez le paramètre Coefficient (c):")

    # Create a text widget to display the result
    results_text_widget = tk.Text(sigmoid_params_window, height=5, width=65)
    results_text_widget.pack(pady=10)
    
    confirm_button = tk.Button(sigmoid_params_window, 
                               text="Calculer Le Résultat", 
                               command=lambda: handle_sigmoid(float(alpha_entry.get()), float(beta_entry.get()), results_text_widget),
                               font=("Arial", 11, "bold"),     
                               bg="#28a745",                  
                               fg="#ffffff",                  
                               activebackground="#218838",   
                               activeforeground="#ffffff",   
                               padx=15, pady=8,               
                               relief="raised",                
                               highlightthickness=0)
    confirm_button.pack(pady=(10, 20))

     # Add a button to return to the home page
    return_button = tk.Button(sigmoid_params_window, 
                              text="Retour à la page d'accueil", 
                              command=sig_return_to_home,
                              font=("Arial", 10, "bold"),  
                              bg="#D87093",         
                              fg="#f0f0f0",             
                              activebackground="#f0f0f0", 
                              activeforeground="#D87093",    
                              padx=10, pady=7,          
                              relief="raised",           
                              highlightthickness=2,     
                              highlightbackground="#a13253")
    return_button.pack(pady=(10, 30))

def handle_sigmoid(alpha, beta, results_text_widget):
    alpha = float(alpha)
    beta = float(beta)
    accuracy = handle_kernel("sigmoid", None, None, alpha, beta)
    y_true, y_pred = get_true_pred("sigmoid", None, None, None, alpha, beta)  # Get true and predicted values
    sigmoid_sensitivity, sigmoid_specificity, sigmoid_tn, sigmoid_fp, sigmoid_fn, sigmoid_tp = calculate_sensitivity_specificity(y_true, y_pred)
    # Function to handle the selected α and β parameters and calculate results for the Sigmoid kernel

    t=test_size

    results_text_widget.delete(1.0, tk.END)

    custom_font = font.Font(family="Helvetica", size=11, weight="bold")

    # Configure a text tag with the custom font
    results_text_widget.tag_configure("bold_font", font=custom_font)

    # Insert text and apply the tag
    results_text_widget.insert(tk.END, "Noyau: Sigmoïde, Précision: ", "bold_font")
    results_text_widget.insert(tk.END, f"{accuracy * 100:.2f}%, ", "bold_font")
    results_text_widget.insert(tk.END, "Alpha: ", "bold_font")
    results_text_widget.insert(tk.END, f"{alpha}, ", "bold_font")
    results_text_widget.insert(tk.END, "Coefficient: ", "bold_font")
    results_text_widget.insert(tk.END, f"{beta}, ", "bold_font")
    results_text_widget.insert(tk.END, "Sensibilité: ", "bold_font")
    results_text_widget.insert(tk.END, f"{sigmoid_sensitivity:.2f}, ", "bold_font")
    results_text_widget.insert(tk.END, "Spécificité: ", "bold_font")
    results_text_widget.insert(tk.END, f"{sigmoid_specificity:.2f}, ", "bold_font")
    # Add similar lines for TN, FP, FN, TP
    results_text_widget.insert(tk.END, "Vrai Négatif: ", "bold_font")
    results_text_widget.insert(tk.END, f"{sigmoid_tn}, ", "bold_font")
    results_text_widget.insert(tk.END, "Faux Positif: ", "bold_font")
    results_text_widget.insert(tk.END, f"{sigmoid_fp}, ", "bold_font")
    results_text_widget.insert(tk.END, "Faux Négatif: ", "bold_font")
    results_text_widget.insert(tk.END, f"{sigmoid_fn}, ", "bold_font")
    results_text_widget.insert(tk.END, "Vrai Positif: ", "bold_font")
    results_text_widget.insert(tk.END, f"{sigmoid_tp}, ", "bold_font")
    results_text_widget.insert(tk.END, "Taille du test: ", "bold_font")
    results_text_widget.insert(tk.END, f"{test_size}\n", "bold_font")

    # Display the result in the provided text widget
    #result_text = f"Noyau: Sigmoïde, Précision: {accuracy * 100:.2f}%, Alpha: {alpha}, Coefficient: {beta}, Taille du test: {test_size}"
    #results_text_widget.delete(1.0, tk.END)  # Clear existing content
    #t=test_size
    #results_text_widget.insert(tk.END, result_text)


def show_linear_parameters():
    # Function to display inputs required for polynomial kernel parameters
    global linear_window
    linear_window = tk.Toplevel(root)
    linear_window.state('zoomed')
    linear_window.configure(bg="#d56392")
    linear_window.iconphoto(False, logo_image)
    linear_window.title("L'affichage Le Résultat Du Noyau Linéaire")
    

    # Create a text widget to display the result
    results_text_widget = tk.Text(linear_window, height=3, width=80)
    results_text_widget.pack(pady=(20, 10))
    
    calculate_button = tk.Button(linear_window, 
                               text="Calculer Le Résultat", 
                               command=lambda: calculate_linear_results(results_text_widget),
                               font=("Arial", 11, "bold"),     
                               bg="#28a745",                  
                               fg="#ffffff",                  
                               activebackground="#218838",   
                               activeforeground="#ffffff",   
                               padx=15, pady=8,               
                               relief="raised",                
                               highlightthickness=0)
    calculate_button.pack(pady=(10, 20))
    
    # Add a button to return to the home page
    return_button = tk.Button(linear_window, 
                              text="Retour à la page d'accueil", 
                              command=lin_return_to_home,
                              font=("Arial", 10, "bold"),  
                              bg="#D87093",         
                              fg="#f0f0f0",             
                              activebackground="#f0f0f0", 
                              activeforeground="#D87093",    
                              padx=10, pady=7,          
                              relief="raised",           
                              highlightthickness=2,     
                              highlightbackground="#a13253")
    return_button.pack(pady=(10, 30))

def calculate_linear_results(results_text_widget):
    accuracy = handle_kernel("linear", None, None, None, None, None)
    y_true, y_pred = get_true_pred("linear", None, None, None, None, None)  # Get true and predicted values
    linear_sensitivity, linear_specificity, linear_tn, linear_fp, linear_fn, linear_tp = calculate_sensitivity_specificity(y_true, y_pred)
    

    # Display the result in the provided text widget
    results_text_widget.delete(1.0, tk.END)  # Clear existing content
    t = test_size  # Unused variable, you might want to remove it

    # Create the font
    custom_font = font.Font(family="Helvetica", size=11, weight="bold")

    # Configure a text tag with the custom font
    results_text_widget.tag_configure("bold_font", font=custom_font)

    # Insert text and apply the tag
    results_text_widget.insert(tk.END, "Noyau: Linéaire, ", "bold_font")  # Corrected line
    results_text_widget.insert(tk.END, "Précision: ", "bold_font")
    results_text_widget.insert(tk.END, f"{accuracy * 100:.2f}%, ", "bold_font")
    results_text_widget.insert(tk.END, "Sensibilité: ", "bold_font")
    results_text_widget.insert(tk.END, f"{linear_sensitivity:.2f}, ", "bold_font")
    results_text_widget.insert(tk.END, "Spécificité: ", "bold_font")
    results_text_widget.insert(tk.END, f"{linear_specificity:.2f}, ", "bold_font")
    # Add similar lines for TN, FP, FN, TP
    results_text_widget.insert(tk.END, "Vrai Négatif: ", "bold_font")
    results_text_widget.insert(tk.END, f"{linear_tn}, ", "bold_font")
    results_text_widget.insert(tk.END, "Faux Positif: ", "bold_font")
    results_text_widget.insert(tk.END, f"{linear_fp}, ", "bold_font")
    results_text_widget.insert(tk.END, "Faux Négatif: ", "bold_font")
    results_text_widget.insert(tk.END, f"{linear_fn}, ", "bold_font")
    results_text_widget.insert(tk.END, "Vrai Positif: ", "bold_font")
    results_text_widget.insert(tk.END, f"{linear_tp}, ", "bold_font")
    results_text_widget.insert(tk.END, "Taille du test: ", "bold_font")
    results_text_widget.insert(tk.END, f"{test_size}\n", "bold_font")

def calculate_sensitivity_specificity(y_true, y_pred):
    # Function to calculate Sensitivity, Specificity, TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity, tn, fp, fn, tp
      
return_button = tk.Button(root, 
                                 text="Retour à la page d'atterissage", 
                                 command=return_to_landing,
                                 font=("Arial", 10, "bold"),  
                                 bg="#D87093",         
                                 fg="#f0f0f0",             
                                 activebackground="#f0f0f0", 
                                 activeforeground="#D87093",    
                                 padx=10, pady=7,          # Increase padding slightly 
                                 relief="raised",           # Add a 3D raised effect
                                 highlightthickness=2,     
                                 highlightbackground="#a13253")
return_button.pack(pady=10)
return_button.place(x=20, y=20)

create_database_button()

test_size_button = tk.Button(root, 
                                 text="La Taille Du Test", 
                                 command=show_test_size_input,
                                 font=("Arial", 11, "bold"),  
                                 bg="#C71585",         
                                 fg="#f0f0f0",             
                                 activebackground="#f0f0f0", 
                                 activeforeground="#C71585",    
                                 padx=15, pady=7,          # Increase padding slightly 
                                 relief="raised",           # Add a 3D raised effect
                                 highlightthickness=2,     
                                 highlightbackground="#a13253")
test_size_button.pack(pady=10)

# Create buttons for SVM kernels
linear_button = tk.Button(root, 
                                 text="Le Noyau Linéaire", 
                                 command=show_linear_parameters,
                                 font=("Arial", 11, "bold"),  
                                 bg="#a13253",         
                                 fg="#f0f0f0",             
                                 activebackground="#f2bbcf", 
                                 activeforeground="#a13253",    
                                 padx=15, pady=7,          # Increase padding slightly 
                                 relief="raised",           # Add a 3D raised effect
                                 highlightthickness=2,     
                                 highlightbackground="#a13253")
linear_button.pack(pady=(10))
linear_button.place(x=400, y=340)

polynomial_button = tk.Button(root, 
                                 text="Le Noyau Polynomial", 
                                 command=show_polynomial_parameters,
                                 font=("Arial", 11, "bold"),  
                                 bg="#a13253",         
                                 fg="#f0f0f0",             
                                 activebackground="#f2bbcf", 
                                 activeforeground="#a13253",    
                                 padx=15, pady=7,          # Increase padding slightly 
                                 relief="raised",           # Add a 3D raised effect
                                 highlightthickness=2,     
                                 highlightbackground="#a13253")
polynomial_button.pack(pady=10)
polynomial_button.place(x=800, y=340)

gaussian_button = tk.Button(root, 
                                 text="Le Noyau Gaussien", 
                                 command=show_gaussian_parameters,
                                 font=("Arial", 11, "bold"),   
                                 bg="#a13253",         
                                 fg="#f0f0f0",             
                                 activebackground="#f2bbcf", 
                                 activeforeground="#a13253",    
                                 padx=15, pady=7,          # Increase padding slightly 
                                 relief="raised",           # Add a 3D raised effect
                                 highlightthickness=2,     
                                 highlightbackground="#a13253")
gaussian_button.pack(pady=10)
gaussian_button.place(x=400, y=440)

sigmoid_button = tk.Button(root, 
                                 text="Le Noyau Sigmoïde", 
                                 command=show_sigmoid_params_window,
                                 font=("Arial", 11, "bold"),  
                                 bg="#a13253",         
                                 fg="#f0f0f0",             
                                 activebackground="#f2bbcf", 
                                 activeforeground="#a13253",    
                                 padx=15, pady=7,          # Increase padding slightly 
                                 relief="raised",           # Add a 3D raised effect
                                 highlightthickness=2,     
                                 highlightbackground="#a13253")
sigmoid_button.pack(pady=10)
sigmoid_button.place(x=800, y=440)

all_kernels_button = tk.Button(root, 
                                 text="Tous Les Noyaux", 
                                 command=show_all_kernel_parameters,
                                 font=("Arial", 12, "bold"),  
                                 bg="#a13253",         
                                 fg="#f0f0f0",             
                                 activebackground="#f2bbcf", 
                                 activeforeground="#a13253",    
                                 padx=30, pady=14,          # Increase padding slightly 
                                 relief="raised",           # Add a 3D raised effect
                                 highlightthickness=2,     
                                 highlightbackground="#a13253")
all_kernels_button.pack(pady=10)
all_kernels_button.place(x=585, y=540)

# Run the main event loop
landing_page.mainloop()                                                                                                                             