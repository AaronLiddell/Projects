#Fitting Gaussians to Co60 histrogram

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Path to the .Spe file (relative to this script)
spe_file = 'Co60/Co60 peaks 2301.Spe'

# Function to parse the .Spe file and extract counts
def read_spe_counts(filename):
	counts = []
	with open(filename, 'r') as f:
		lines = f.readlines()
		data_section = False
		skip_next = False
		for line in lines:
			if line.strip() == '$DATA:':
				data_section = True
				skip_next = True  # skip the channel range line
				continue
			if data_section:
				if line.strip().startswith('$'):
					break
				if skip_next:
					skip_next = False
					continue
				try:
					value = int(line.strip())
					counts.append(value)
				except ValueError:
					continue
	return np.array(counts)


# Read counts from the .Spe file
counts = read_spe_counts(spe_file)
if counts.size == 0:
	print("Warning: No counts data found in the .Spe file. Please check the file format.")

# Chop off the first 50 bins for better visualization and fitting
chop_bins = 50
counts = counts[chop_bins:]


# Function to parse ROI values from the .Spe file
def read_spe_rois(filename, chop_bins=0):
	roi_list = []
	with open(filename, 'r') as f:
		lines = f.readlines()
		for i, line in enumerate(lines):
			if line.strip() == '$ROI:':
				try:
					n_rois = int(lines[i+1].strip())
					roi_values = lines[i+2:i+2+n_rois]
					for roi_line in roi_values:
						parts = roi_line.strip().split()
						if len(parts) == 2:
							start, end = map(int, parts)
							roi_list.append((start - chop_bins, end - chop_bins))
				except Exception as e:
					print(f"Error parsing ROI: {e}")
				break
	return roi_list

# Parse ROIs from the .Spe file
roi_list = read_spe_rois(spe_file, chop_bins)

# --- Gaussian fitting section ---

# Define a Gaussian function
def gaussian(x, a, mu, sigma, c):
	return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + c


x = np.arange(len(counts))

plt.figure(figsize=(10, 6))
plt.plot(x, counts, label='Spectrum')

fit_results = []
for i, (start, end) in enumerate(roi_list):
	x_roi = x[start:end+1]
	y_roi = counts[start:end+1]
	# Initial guess: amplitude, mean, stddev, background
	a0 = y_roi.max() - y_roi.min() if y_roi.size > 0 else 1
	mu0 = x_roi[np.argmax(y_roi)] if y_roi.size > 0 else (start + end) / 2
	sigma0 = (end - start) / 6
	c0 = y_roi.min() if y_roi.size > 0 else 0
	p0 = [a0, mu0, sigma0, c0]
	try:
		if y_roi.size > 0:
			popt, pcov = curve_fit(gaussian, x_roi, y_roi, p0=p0)
			fit_results.append((popt, np.sqrt(np.diag(pcov))))
			plt.plot(x_roi, gaussian(x_roi, *popt), '--', label=f'Gaussian Fit {i+1}')
		else:
			print(f"ROI {i+1} is empty, skipping fit.")
	except Exception as e:
		print(f"Fit failed for ROI {i+1}: {e}")


plt.xlabel('Channel')

# Print fit results, FWHM, and total events in each peak
for i, ((params, errors), (start, end)) in enumerate(zip(fit_results, roi_list)):
	a, mu, sigma, c = params
	a_err, mu_err, sigma_err, c_err = errors
	# FWHM for a Gaussian: FWHM = 2.3548 * sigma
	fwhm = 2.3548 * sigma
	fwhm_err = 2.3548 * sigma_err
	# Total events in the ROI (sum of counts in the region)
	total_events = np.sum(counts[start:end+1])
	print(f"Peak {i+1} fit parameters:")
	print(f"  Amplitude: {a:.2f} ± {a_err:.2f}")
	print(f"  Mean (channel): {mu:.2f} ± {mu_err:.2f}")
	print(f"  Sigma: {sigma:.2f} ± {sigma_err:.2f}")
	print(f"  Background: {c:.2f} ± {c_err:.2f}")
	print(f"  FWHM: {fwhm:.2f} ± {fwhm_err:.2f} (channels)")
	print(f"  Total events in ROI: {total_events}")
plt.ylabel('Counts')
plt.title('Co60 Spectrum with Gaussian Fits')
plt.legend()
plt.show()

# Print fit results
#for i, (params, errors) in enumerate(fit_results):
#	print(f"Peak {i+1} fit parameters:")
#	print(f"  Amplitude: {params[0]:.2f} ± {errors[0]:.2f}")
#	print(f"  Mean (channel): {params[1]:.2f} ± {errors[1]:.2f}")
#	print(f"  Sigma: {params[2]:.2f} ± {errors[2]:.2f}")
#	print(f"  Background: {params[3]:.2f} ± {errors[3]:.2f}")
