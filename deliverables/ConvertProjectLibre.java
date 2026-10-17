import java.io.FileInputStream;
import com.projectlibre1.exchange.LocalFileImporter;
import com.projectlibre1.exchange.MicrosoftImporter;
import com.projectlibre1.pm.task.Project;
import com.projectlibre1.pm.task.ProjectFactory;
import com.projectlibre1.util.Environment;
import com.projectlibre1.job.JobQueue;
import com.projectlibre1.session.SessionFactory;
import com.projectlibre1.pm.task.Task;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import com.projectlibre1.configuration.Configuration;
import com.projectlibre1.field.Field;

public class ConvertProjectLibre {
    private static final Pattern RACI = Pattern.compile(
            "RACI: R: (.*?); A: (.*?); C: (.*?); I: (.*?)\\.$");

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            throw new IllegalArgumentException("Usage: ConvertProjectLibre input.xml output.pod");
        }
        System.setProperty("java.awt.headless", "true");
        Environment.setStandAlone(true);
        SessionFactory.getInstance().setJobQueue(new JobQueue("conversion", true));
        MicrosoftImporter input = new MicrosoftImporter();
        Project target = ProjectFactory.getInstance().createProject();
        if (target == null) throw new IllegalStateException("ProjectFactory returned no project");
        input.setProject(target);
        Project project;
        try (FileInputStream stream = new FileInputStream(args[0])) {
            project = input.loadProject(stream);
        }
        if (project == null) {
            throw new IllegalStateException("ProjectLibre returned no project");
        }
        for (Object item : project.getTasks()) {
            Task task = (Task) item;
            Matcher match = RACI.matcher(task.getNotes());
            if (!match.find()) {
                throw new IllegalStateException("Missing RACI for " + task.getName());
            }
            for (int field = 0; field < 4; field++) {
                task.setCustomText(field, match.group(field + 1));
            }
        }
        LocalFileImporter output = new LocalFileImporter();
        output.setProject(project);
        output.setFileName(args[1]);
        output.exportFile();
        LocalFileImporter check = new LocalFileImporter();
        check.setFileName(args[1]);
        check.importFile();
        Project reopened = check.getProject();
        if (reopened == null) throw new IllegalStateException("Could not reopen POD");
        int tasks = 0;
        int milestones = 0;
        int dependencies = 0;
        for (Object item : reopened.getTasks()) {
            Task task = (Task) item;
            tasks++;
            if (task.isMilestone()) milestones++;
            dependencies += task.getPredecessorList().size();
            for (int field = 0; field < 4; field++) {
                if (task.getCustomText(field) == null) {
                    throw new IllegalStateException("Missing RACI Text" + (field + 1) +
                            " for " + task.getName());
                }
                Field visibleField = Configuration.getFieldFromId("Field.customText" + (field + 1));
                if (visibleField == null || !task.getCustomText(field).equals(visibleField.getValue(task))) {
                    throw new IllegalStateException("ProjectLibre Text" + (field + 1) +
                            " column does not match RACI for " + task.getName());
                }
            }
        }
        if (tasks != 25 || milestones != 5 || dependencies != 34) {
            throw new IllegalStateException("POD content mismatch: tasks=" + tasks +
                    ", milestones=" + milestones + ", dependencies=" + dependencies);
        }
        System.out.println("Verified POD: 25 tasks, 5 milestones, 34 dependencies, RACI in Text1-Text4: " + args[1]);
    }
}
