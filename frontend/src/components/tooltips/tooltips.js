const edit = `

Here is the File editing page. Here is the returned image from the file editor.
Beside the image there are three options for the different commands; Users may either make the areas transparant, completely black or can also restore the area to base images state.
The toolbar also allows you to reset the image to the original uploaded version.
<br />
While hovering over the image and a mask, you have 4 options:
<br />

Apply to Rest Of Image
This applies the given rule to everywhere except the current are being highlighted.
<br />

Apply to Highlighted Area Only.
This simply applies the current rule to the are being highlighted only.
<br />

Use Touch Up Tool
This is a tool that follows the cursor and on left click applies the given rule to the red highlighted are. The area size may be increased or decreased via the toolbar while active. To exit the too, simply right click.
<br />

Save
This downloads the image as the specified file type and marks the file into the saved category on the view files page.
`
const upload = `
This page accepts, jpeg, jpg, tif, tiff, png and zip/ tar formats only.
`
const filePage = `
This is where you can see all the files in their different edit states. Files in all stages may be deleted or shared with other users in their respective states. 
<br />
Uploaded
In Uploaded files may be viewed or processed with or without a prompt. If there is no prompt no editing may occur so all generated masks will be returned along with the image for manual editing. If more than 50 files are selected at once, it may take some time for the files to leave this state.

<br />
Processing
Shows the files currently being processed, these are all done in batches up of to 50 and move onto processed once complete.
<br />
Processed
Here you may view edit & save the image that has been returned from the image processor.
<br />
Saved
Files here may be dowloaded individually, or if multiple are selected they will downloaded into zip/tar formats. Files can also be re processed here again with prompts or without. 
`

export default {
    edit,
    upload,
    filePage
}