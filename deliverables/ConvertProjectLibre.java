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
import com.projectlibre1.datatype.Duration;
import com.projectlibre1.datatype.TimeUnit;
import java.util.Date;
import java.text.SimpleDateFormat;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.HashSet;
import java.util.Set;
import com.projectlibre1.pm.calendar.WorkingCalendar;
import com.projectlibre1.pm.calendar.WorkDay;

public class ConvertProjectLibre {
    private static final Pattern RACI = Pattern.compile(
            "RACI: R: (.*?); A: (.*?); C: (.*?); I: (.*?)\\.$");
    private static final Pattern EXECUTOR = Pattern.compile("исполнитель (.*?); выход:");

    private static String tag(Element parent, String name) {
        NodeList elements = parent.getElementsByTagNameNS("http://schemas.microsoft.com/project", name);
        return elements.getLength() == 0 ? "" : elements.item(0).getTextContent();
    }

    private static Map<String, Integer> plannedDays(String fileName) throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setNamespaceAware(true);
        factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
        NodeList tasks = factory.newDocumentBuilder().parse(new File(fileName))
                .getElementsByTagNameNS("http://schemas.microsoft.com/project", "Task");
        Map<String, Integer> result = new HashMap<>();
        for (int i = 0; i < tasks.getLength(); i++) {
            Element task = (Element) tasks.item(i);
            Matcher duration = Pattern.compile("PT(\\d+)H0M0S").matcher(tag(task, "Duration"));
            if (!duration.matches()) throw new IllegalStateException("Unexpected duration for " + tag(task, "WBS"));
            int hours = Integer.parseInt(duration.group(1));
            if (hours % 8 != 0) throw new IllegalStateException("Duration is not in workdays");
            result.put(tag(task, "WBS"), hours / 8);
        }
        return result;
    }

    private static Map<String, String[]> plannedDates(String fileName) throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setNamespaceAware(true);
        factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
        NodeList tasks = factory.newDocumentBuilder().parse(new File(fileName))
                .getElementsByTagNameNS("http://schemas.microsoft.com/project", "Task");
        Map<String, String[]> result = new HashMap<>();
        for (int i = 0; i < tasks.getLength(); i++) {
            Element task = (Element) tasks.item(i);
            result.put(tag(task, "WBS"), new String[] {
                    tag(task, "Start").substring(0, 10), tag(task, "Finish").substring(0, 10)
            });
        }
        return result;
    }

    private static void applyCalendarExceptions(Project project, String fileName) throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setNamespaceAware(true);
        factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
        NodeList exceptions = factory.newDocumentBuilder().parse(new File(fileName))
                .getElementsByTagNameNS("http://schemas.microsoft.com/project", "Exception");
        Set<WorkingCalendar> calendars = new HashSet<>();
        calendars.add((WorkingCalendar) project.getEffectiveWorkCalendar());
        for (Object item : project.getTasks())
            calendars.add((WorkingCalendar) ((Task) item).getEffectiveWorkCalendar());
        for (int i = 0; i < exceptions.getLength(); i++) {
            Element exception = (Element) exceptions.item(i);
            if (!tag(exception, "DayWorking").equals("0")) continue;
            long day = LocalDate.parse(tag(exception, "FromDate").substring(0, 10))
                    .atStartOfDay(ZoneOffset.UTC).toInstant().toEpochMilli();
            for (WorkingCalendar calendar : calendars) {
                WorkDay nonWorking = new WorkDay(day);
                nonWorking.getWorkingHours().setNonWorking();
                calendar.addOrReplaceException(nonWorking);
                calendar.invalidate();
                calendar.notifyChanged();
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            throw new IllegalArgumentException("Usage: ConvertProjectLibre input.xml output.pod");
        }
        System.setProperty("java.awt.headless", "true");
        Environment.setStandAlone(true);
        SessionFactory.getInstance().setJobQueue(new JobQueue("conversion", true));
        if (args[0].equals("--inspect")) {
            LocalFileImporter existing = new LocalFileImporter();
            existing.setFileName(args[1]);
            existing.importFile();
            Project compared = existing.getProject();
            if (compared == null) throw new IllegalStateException("Could not open POD");
            int index = 0;
            for (Object item : compared.getTasks()) {
                Task task = (Task) item;
                index++;
                if (index <= 7 || index > compared.getTasks().size() - 5)
                    System.out.println(index + " " + task.getWbs() + " " + task.getName() +
                            " days=" + Duration.getValue(task.getRawDuration()) +
                            " start=" + new Date(task.getStart()) + " end=" + new Date(task.getEnd()));
            }
            System.out.println("POD tasks=" + index + " projectEnd=" + new Date(compared.getEnd()));
            return;
        }
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
        Map<String, Integer> daysByWbs = plannedDays(args[0]);
        Map<String, String[]> datesByWbs = plannedDates(args[0]);
        applyCalendarExceptions(project, args[0]);
        for (Object item : project.getTasks()) {
            Task task = (Task) item;
            ((com.projectlibre1.pm.task.NormalTask) task).invalidateAssignmentCalendars();
            Integer days = daysByWbs.get(task.getWbs());
            if (days == null) throw new IllegalStateException("Unknown WBS " + task.getWbs());
            if (days > 0) ((com.projectlibre1.pm.task.NormalTask) task)
                    .setDuration(Duration.getInstance(days, TimeUnit.DAYS));
            Matcher match = RACI.matcher(task.getNotes());
            if (!match.find()) {
                throw new IllegalStateException("Missing RACI for " + task.getName());
            }
            for (int field = 0; field < 4; field++) {
                task.setCustomText(field, match.group(field + 1));
            }
            Matcher executor = EXECUTOR.matcher(task.getNotes());
            if (!executor.find()) throw new IllegalStateException("Missing executor role for " + task.getName());
            task.setCustomText(4, executor.group(1));
        }
        project.recalculate();
        // ProjectLibre can retain a nonworking day as the start of a one-day task.
        // Move only such starts to the next working interval after a calendar exception.
        for (Object item : project.getTasks()) {
            Task task = (Task) item;
            if (task.isMilestone()) continue;
            String expectedStart = datesByWbs.get(task.getWbs())[0];
            String actualStart = new SimpleDateFormat("yyyy-MM-dd").format(new Date(task.getStart()));
            String actualEnd = new SimpleDateFormat("yyyy-MM-dd").format(new Date(task.getEnd()));
            if (actualStart.compareTo(expectedStart) < 0 &&
                    actualEnd.equals(datesByWbs.get(task.getWbs())[1])) {
                long earliest = LocalDate.parse(expectedStart).atTime(9, 0)
                        .toInstant(ZoneOffset.UTC).toEpochMilli();
                task.setStartNoEarlierThan(earliest);
            }
        }
        project.recalculate();
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
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
        for (Object item : reopened.getTasks()) {
            Task task = (Task) item;
            tasks++;
            if (task.isMilestone()) milestones++;
            dependencies += task.getPredecessorList().size();
            int expectedDays = daysByWbs.get(task.getWbs());
            if (Duration.getValue(task.getRawDuration()) != expectedDays) {
                throw new IllegalStateException("Wrong duration for " + task.getWbs() + ": " +
                        Duration.getValue(task.getRawDuration()) + " instead of " + expectedDays);
            }
            String[] expectedDates = datesByWbs.get(task.getWbs());
            if (!dateFormat.format(new Date(task.getStart())).equals(expectedDates[0]) ||
                    !dateFormat.format(new Date(task.getEnd())).equals(expectedDates[1])) {
                throw new IllegalStateException("Wrong schedule dates for " + task.getWbs() +
                        ": " + dateFormat.format(new Date(task.getStart())) + ".." + dateFormat.format(new Date(task.getEnd())) +
                        " expected " + expectedDates[0] + ".." + expectedDates[1]);
            }
            for (int field = 0; field < 5; field++) {
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
        System.out.println("Verified POD: 20 works, 5 milestones, 34 dependencies, dates match XML, RACI in Text1-Text4, executor role in Text5: " + args[1]);
    }
}
