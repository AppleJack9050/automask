- To run api -> cd ./api and run uvicorn endpoints:app --reload

- To run frontend Vue cd ./automask-frontend 
- If you do not have node installed please do
- Run npm i (or windows equivalent) and then run npm run dev (this is only temporary before I dockerize and can deploy properly)

### To Use App

- To upload files go to input and drag and drop as many files as you want
- Next go to View Files and click process files (for now this just puts all of them together so just need to click once)
- after a few seconds they should all move to processed then click edit, which should bring you to the editfile where the fun happens

- (Sometimes here it can get stuck for some reason so theres a console.log in EditFile.vue ~line 265 if you change it it should load )

- To remove an object rightclick -> remove from image
- To select only the object rightclick -> select
- To save rightclick -> save and a modal will appear with file type
