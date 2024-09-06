all:
	@rm -rf output
	@docker build -t python-plot . 
	@echo "Running...\n"
	@docker run -v ./output:/output python-plot

clean:
	@docker system prune -af
	@rm -rf output